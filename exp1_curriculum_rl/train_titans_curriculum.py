"""
Experiment 1.5: Staged Curriculum Learning (SFT -> GRPO) for Small Titans (MAC).

Implements the 3-level "Mini Study Loop" designed for Kaggle free-tier GPUs (T4/P100 16GB):
  - Level 1: Single-hop search + logical structuring
  - Level 2: Multi-hop concept decomposition (find prerequisite unknown concept -> search -> combine)
  - Level 3: Memory reuse (recall a concept searched earlier in the same session without re-searching)

Training Pipeline:
  - Stage 1 (SFT): Masked cross-entropy over structured <think>/<search>/<answer> demonstrations
    (prompt and environment <result> tokens are masked out).
  - Stage 2 (GRPO): Group Relative Policy Optimization with live mock-search rollouts,
    automatic format/accuracy/memory-reuse rewards, KL penalty against frozen SFT reference,
    and per-rollout gradient accumulation to prevent OOM on 16GB VRAM.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Special Tags & Self-Contained Character Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|unk|>",
    "<think>",
    "</think>",
    "<search>",
    "</search>",
    "<result>",
    "</result>",
    "<answer>",
    "</answer>",
]

# Concept pool: enough variety that the model cannot memorize fixed answers
CONCEPT_NAMES = [
    "アルファ", "ベータ", "ガンマ", "デルタ", "シグマ", "オメガ",
    "カッパ", "ラムダ", "ゼータ", "イータ", "ミュー", "ニュー",
    "パイ", "ロー", "タウ", "ファイ", "カイ", "プサイ",
    "ノードA", "ノードB", "ノードC", "ノードD", "ノードE", "ノードF",
    "係数1", "係数2", "係数3", "係数4", "係数5", "係数6",
    "定数A", "定数B", "定数C", "定数D", "変数P", "変数Q",
]

DISTRACTOR_TURNS = [
    "メモ: 途中経過を保持中。",
    "メモ: 次の確認事項を待機中。",
    "メモ: 前の定義を記憶しておく。",
    "メモ: 文脈を継続して整理する。",
]


class CurriculumTokenizer:
    """Deterministic special-tag + character tokenizer (zero external downloads)."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<|pad|>"]
        self.eos_token_id = self.token_to_id["<|eos|>"]
        self.unk_token_id = self.token_to_id["<|unk|>"]
        self.search_end_id = self.token_to_id["</search>"]
        self.vocab_size = len(self.token_to_id)

        # Regex that matches special tags first, then individual characters
        tag_pattern = "|".join(re.escape(t) for t in SPECIAL_TOKENS)
        self._split_re = re.compile(f"({tag_pattern})")

    @classmethod
    def build_default(cls) -> "CurriculumTokenizer":
        chars: set[str] = set()
        # Base ASCII digits, letters, math symbols, punctuation, whitespace
        for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-*=():.,!? \n":
            chars.add(ch)
        # Japanese characters used across prompts, templates, concepts, distractors
        sample_texts = [
            "質問: のとき、はいくつ？",
            "目的: を求める。既知: 不明な概念: 整理: にはが必要。記憶: よって",
            "エラー: 該当なし",
            "前回",
            *CONCEPT_NAMES,
            *DISTRACTOR_TURNS,
        ]
        for s in sample_texts:
            for ch in s:
                chars.add(ch)

        vocab: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            vocab[tok] = len(vocab)
        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self._split_re.split(text):
            if not part:
                continue
            if part in self.token_to_id:
                ids.append(self.token_to_id[part])
            else:
                for ch in part:
                    ids.append(self.token_to_id.get(ch, self.unk_token_id))
        return ids

    def decode(self, ids: list[int], skip_special_pad: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if skip_special_pad and tok == "<|pad|>":
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CurriculumTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in data.items()})


# ---------------------------------------------------------------------------
# 2. Procedural 3-Level Curriculum Generator & Mock Search Environment
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    level: int
    prompt: str
    kb: dict[str, str]
    expected_answer: int
    ideal_completion: str
    memory_reuse_target: bool = False  # True on Level 3 Turn 2 (no search expected)


def _apply_op(val: int, op: str, k: int) -> int:
    if op == "+":
        return val + k
    if op == "-":
        return val - k
    return val * k


def generate_episode(level: int, rng: random.Random) -> Episode:
    """Generates a fresh episode with an episode-local knowledge base."""
    assert level in (1, 2, 3), f"Invalid level: {level}"

    if level == 1:
        c1 = rng.choice(CONCEPT_NAMES)
        x = rng.randint(2, 15)
        op = rng.choice(["+", "*"])
        k = rng.randint(2, 9) if op == "+" else rng.randint(2, 5)
        ans = _apply_op(x, op, k)
        rule_c1 = f"{c1} = X {op} {k}"
        kb = {c1: rule_c1}
        prompt = f"質問: X={x} のとき、{c1} はいくつ？\n"
        ideal = (
            f"<think>目的: {c1}を求める。既知: X={x}。不明な概念: {c1}</think>"
            f"<search>{c1}</search>"
            f"<result>{rule_c1}</result>"
            f"<think>整理: {x} {op} {k} = {ans}</think>"
            f"<answer>{ans}</answer><|eos|>"
        )
        return Episode(level=1, prompt=prompt, kb=kb, expected_answer=ans, ideal_completion=ideal)

    if level == 2:
        c1, c2 = rng.sample(CONCEPT_NAMES, 2)
        x = rng.randint(2, 10)
        op1 = rng.choice(["+", "*"])
        k1 = rng.randint(2, 8) if op1 == "+" else rng.randint(2, 4)
        v1 = _apply_op(x, op1, k1)

        op2 = rng.choice(["+", "*"])
        k2 = rng.randint(2, 8) if op2 == "+" else rng.randint(2, 4)
        ans = _apply_op(v1, op2, k2)

        rule_c1 = f"{c1} = X {op1} {k1}"
        rule_c2 = f"{c2} = {c1} {op2} {k2}"
        kb = {c1: rule_c1, c2: rule_c2}
        prompt = f"質問: X={x} のとき、{c2} はいくつ？\n"
        ideal = (
            f"<think>目的: {c2}を求める。既知: X={x}。不明な概念: {c2}</think>"
            f"<search>{c2}</search>"
            f"<result>{rule_c2}</result>"
            f"<think>整理: {c2}には{c1}が必要。不明な概念: {c1}</think>"
            f"<search>{c1}</search>"
            f"<result>{rule_c1}</result>"
            f"<think>整理: {c1} = {x} {op1} {k1} = {v1}。{c2} = {v1} {op2} {k2} = {ans}</think>"
            f"<answer>{ans}</answer><|eos|>"
        )
        return Episode(level=2, prompt=prompt, kb=kb, expected_answer=ans, ideal_completion=ideal)

    # Level 3: Memory reuse (Turn 1 is already in the conversation stream; Turn 2 asks again with new X)
    c1 = rng.choice(CONCEPT_NAMES)
    x1 = rng.randint(2, 12)
    x2 = rng.randint(2, 12)
    while x2 == x1:
        x2 = rng.randint(2, 12)
    op = rng.choice(["+", "*"])
    k = rng.randint(2, 9) if op == "+" else rng.randint(2, 5)
    ans1 = _apply_op(x1, op, k)
    ans2 = _apply_op(x2, op, k)
    rule_c1 = f"{c1} = X {op} {k}"
    kb = {c1: rule_c1}

    distractor = rng.choice(DISTRACTOR_TURNS)
    turn1 = (
        f"質問: X={x1} のとき、{c1} はいくつ？\n"
        f"<think>目的: {c1}を求める。既知: X={x1}。不明な概念: {c1}</think>"
        f"<search>{c1}</search>"
        f"<result>{rule_c1}</result>"
        f"<think>整理: {x1} {op} {k} = {ans1}</think>"
        f"<answer>{ans1}</answer>\n"
        f"{distractor}\n"
    )
    prompt = f"{turn1}質問: X={x2} のとき、{c1} はいくつ？\n"
    ideal = (
        f"<think>目的: {c1}を求める。既知: X={x2}。記憶: {rule_c1}。整理: {x2} {op} {k} = {ans2}</think>"
        f"<answer>{ans2}</answer><|eos|>"
    )
    return Episode(
        level=3,
        prompt=prompt,
        kb=kb,
        expected_answer=ans2,
        ideal_completion=ideal,
        memory_reuse_target=True,
    )


def tokenize_sft_episode(
    ep: Episode,
    tokenizer: CurriculumTokenizer,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenizes an SFT episode and builds masked labels:
      - Prompt tokens -> -100
      - Environment <result>...</result> tokens -> -100
      - Model-generated tokens (<think>, <search>, <answer>, <|eos|>) -> target token ID
    """
    prompt_ids = tokenizer.encode(ep.prompt)
    comp_ids = tokenizer.encode(ep.ideal_completion)

    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + list(comp_ids)

    # Mask out <result> ... </result> inside completion (environment output)
    res_start = tokenizer.token_to_id["<result>"]
    res_end = tokenizer.token_to_id["</result>"]
    in_result = False
    for idx in range(len(prompt_ids), len(labels)):
        tok_id = input_ids[idx]
        if tok_id == res_start:
            in_result = True
            labels[idx] = -100
        elif in_result:
            labels[idx] = -100
            if tok_id == res_end:
                in_result = False

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return input_ids, labels


# ---------------------------------------------------------------------------
# 3. Models: Titans (MAC) + Fallback TinyTransformer for Local Smoke Tests
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    dim: int
    depth: int
    segment_len: int
    num_persist_mem_tokens: int
    num_longterm_mem_tokens: int
    seq_len: int
    sft_batch_size: int
    sft_grad_accum: int
    sft_lr: float
    grpo_lr: float


PRESETS: dict[str, ModelPreset] = {
    # Ultra-fast CPU/GPU smoke test (~1.5M params)
    "micro": ModelPreset(
        dim=128,
        depth=3,
        segment_len=32,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=8,
        seq_len=320,
        sft_batch_size=8,
        sft_grad_accum=2,
        sft_lr=1.5e-3,
        grpo_lr=1e-4,
    ),
    # Kaggle T4/P100 fast curriculum run (~6M params, <3GB VRAM)
    "tiny": ModelPreset(
        dim=256,
        depth=4,
        segment_len=64,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
        seq_len=384,
        sft_batch_size=4,
        sft_grad_accum=8,
        sft_lr=8e-4,
        grpo_lr=5e-5,
    ),
    # Kaggle T4/P100 higher-capacity run (~16M params, OOM-safe with batch_size=2)
    "small": ModelPreset(
        dim=384,
        depth=6,
        segment_len=64,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        seq_len=384,
        sft_batch_size=2,
        sft_grad_accum=16,
        sft_lr=5e-4,
        grpo_lr=3e-5,
    ),
}


class FallbackCausalTransformer(nn.Module):
    """Minimal causal Transformer used only when --model_type tiny_transformer is requested."""

    def __init__(self, num_tokens: int, dim: int, depth: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        nhead = max(1, dim // 64)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = x.shape
        pos = torch.arange(seqlen, device=x.device).unsqueeze(0).expand(bsz, seqlen)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=causal_mask)
        return self.head(self.ln_f(h))


def build_model(model_type: str, model_cfg: dict[str, Any], device: torch.device) -> nn.Module:
    if model_type == "titans":
        from titans_pytorch import MemoryAsContextTransformer

        model = MemoryAsContextTransformer(
            num_tokens=model_cfg["num_tokens"],
            dim=model_cfg["dim"],
            depth=model_cfg["depth"],
            segment_len=model_cfg["segment_len"],
            num_persist_mem_tokens=model_cfg["num_persist_mem_tokens"],
            num_longterm_mem_tokens=model_cfg["num_longterm_mem_tokens"],
        )
        return model.to(device)

    model = FallbackCausalTransformer(
        num_tokens=model_cfg["num_tokens"],
        dim=model_cfg["dim"],
        depth=model_cfg["depth"],
        max_seq_len=max(512, model_cfg.get("seq_len", 384) + 128),
    )
    return model.to(device)


def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        major, _ = torch.cuda.get_device_capability()
        if major >= 8 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif major >= 7:
            dtype = torch.float16
        else:
            dtype = torch.float32
        return device, dtype
    return torch.device("cpu"), torch.float32


# ---------------------------------------------------------------------------
# 4. Interactive Mock-Search Rollout & Reward Function
# ---------------------------------------------------------------------------

SEARCH_RE = re.compile(r"<search>([^<>]+)</search>")
ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class RolloutResult:
    full_ids: list[int]
    action_mask: list[int]  # 1 = model-generated token, 0 = prompt or <result>
    completion_text: str
    searches_made: list[str]
    reward: float
    correct: bool
    format_ok: bool


def score_rollout(ep: Episode, completion_text: str, searches_made: list[str]) -> tuple[float, bool, bool]:
    """Automatic verifier for format, answer accuracy, search hygiene, and memory reuse."""
    has_think = bool(THINK_RE.search(completion_text))
    ans_match = ANSWER_RE.search(completion_text)
    format_ok = has_think and (ans_match is not None)

    reward = 0.0
    if format_ok:
        reward += 0.2

    correct = False
    if ans_match is not None:
        try:
            pred = int(ans_match.group(1))
            if pred == ep.expected_answer:
                correct = True
                reward += 1.0
        except ValueError:
            pass

    # Penalize invalid searches or excessive search loops
    invalid_searches = sum(1 for q in searches_made if q not in ep.kb)
    if invalid_searches > 0:
        reward -= 0.1 * invalid_searches
    if len(searches_made) > 3:
        reward -= 0.2

    # Level 3 memory-reuse bonus: answered correctly without re-searching!
    if ep.memory_reuse_target and correct and len(searches_made) == 0:
        reward += 0.3

    return reward, correct, format_ok


@torch.no_grad()
def run_interactive_rollout(
    model: nn.Module,
    tokenizer: CurriculumTokenizer,
    ep: Episode,
    device: torch.device,
    max_new_tokens: int = 160,
    max_seq_len: int = 384,
    temperature: float = 0.8,
    top_p: float = 0.9,
    greedy: bool = False,
    max_searches: int = 3,
) -> RolloutResult:
    """
    Autoregressively generates a completion while intercepting <search>...</search>
    and injecting <result>...</result> from the episode's mock search KB.
    """
    model.eval()
    prompt_ids = tokenizer.encode(ep.prompt)
    seq = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    searches_made: list[str] = []

    steps = 0
    while steps < max_new_tokens and len(seq) < max_seq_len:
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]  # (1, V)

        if greedy or temperature <= 1e-5:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_probs[remove_mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
                next_id = int(sorted_idx.gather(-1, sampled_pos).item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        seq.append(next_id)
        action_mask.append(1)
        steps += 1

        if next_id == tokenizer.eos_token_id:
            break

        # Intercept </search> and inject environment <result>...</result>
        if next_id == tokenizer.search_end_id and len(searches_made) < max_searches:
            comp_so_far = tokenizer.decode(seq[len(prompt_ids):])
            matches = SEARCH_RE.findall(comp_so_far)
            if matches:
                query = matches[-1].strip()
                searches_made.append(query)
                result_str = ep.kb.get(query, "エラー: 該当なし")
                env_ids = tokenizer.encode(f"<result>{result_str}</result>")
                if len(seq) + len(env_ids) < max_seq_len:
                    seq.extend(env_ids)
                    action_mask.extend([0] * len(env_ids))

    completion_text = tokenizer.decode(seq[len(prompt_ids):])
    reward, correct, format_ok = score_rollout(ep, completion_text, searches_made)
    return RolloutResult(
        full_ids=seq,
        action_mask=action_mask,
        completion_text=completion_text,
        searches_made=searches_made,
        reward=reward,
        correct=correct,
        format_ok=format_ok,
    )


# ---------------------------------------------------------------------------
# 5. Stage 1: Curriculum SFT (Foundation Building)
# ---------------------------------------------------------------------------

def sample_curriculum_level(progress: float, max_level: int = 3) -> int:
    """
    Smoothly shifts level distribution from Level 1 -> Level 2 -> Level 3 as training progresses,
    while always retaining a replay ratio of earlier levels to avoid forgetting.
    """
    if max_level == 1:
        return 1
    r = random.random()
    if progress < 0.35:
        return 1 if r < 0.80 else 2
    if progress < 0.70 or max_level == 2:
        if r < 0.30:
            return 1
        if r < 0.85 or max_level == 2:
            return 2
        return 3
    # Late stage: mix all 3 levels with emphasis on Level 2 & 3
    if r < 0.20:
        return 1
    if r < 0.55:
        return 2
    return 3


def collate_sft_batch(
    episodes: list[tuple[list[int], list[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in episodes)
    bsz = len(episodes)
    x = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, lbls) in enumerate(episodes):
        n = len(ids)
        x[i, :n] = torch.tensor(ids, dtype=torch.long)
        y[i, :n] = torch.tensor(lbls, dtype=torch.long)
    return x, y


def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    p = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return max_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def run_sft_stage(
    model: nn.Module,
    tokenizer: CurriculumTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    max_lr: float,
    output_dir: Path,
    log_every: int = 20,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    use_amp = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_scaler)
    )
    warmup = max(1, min(100, steps // 10))

    model.train()
    t0 = time.time()
    print(f"\n[Stage 1: SFT] steps={steps}, batch={batch_size}x{grad_accum}, seq_len={seq_len}, lr={max_lr}")

    for step in range(1, steps + 1):
        progress = (step - 1) / max(1, steps)
        lr = cosine_lr(step - 1, warmup, steps, max_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            batch_items = [
                tokenize_sft_episode(
                    generate_episode(sample_curriculum_level(progress, max_level=3), rng),
                    tokenizer,
                    max_seq_len=seq_len,
                )
                for _ in range(batch_size)
            ]
            x, y = collate_sft_batch(batch_items, tokenizer.pad_token_id)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(x[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                scaled_loss = loss / grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accum_loss += float(loss.item()) / grad_accum

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % log_every == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            print(f"  [SFT] step {step:5d}/{steps} | loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

    ckpt_path = output_dir / "stage1_sft_latest.pt"
    torch.save({"stage": "sft", "steps": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"[Stage 1: SFT] Saved checkpoint -> {ckpt_path}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 6. Stage 2: GRPO (Error-Aware Reinforcement Learning)
# ---------------------------------------------------------------------------

def sequence_token_logprobs(
    model: nn.Module,
    full_ids: list[int],
    action_mask: list[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (token_logprobs, mask) aligned to shifted targets (length T-1).
    Only positions where mask == 1 correspond to model-generated actions.
    """
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    mask = torch.tensor([action_mask[1:]], dtype=torch.float32, device=device)

    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    with amp_ctx:
        logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chosen_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    return chosen_lp.squeeze(0), mask.squeeze(0)


def run_grpo_stage(
    model: nn.Module,
    tokenizer: CurriculumTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int,
    group_size: int,
    max_new_tokens: int,
    seq_len: int,
    lr: float,
    kl_beta: float,
    output_dir: Path,
    log_every: int = 5,
    seed: int = 1000,
) -> None:
    """
    OOM-safe GRPO:
      1. Freeze a reference copy of the SFT model for KL regularization.
      2. For each step, sample an episode (curriculum Level 1 -> 2 -> 3) and generate G rollouts.
      3. Compute group-relative advantages: (r_i - mean) / (std + 1e-6).
      4. Accumulate gradients ONE rollout at a time so peak VRAM stays at batch_size = 1.
    """
    rng = random.Random(seed)
    use_amp = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    t0 = time.time()
    print(
        f"\n[Stage 2: GRPO] steps={steps}, group_size={group_size}, "
        f"max_new_tokens={max_new_tokens}, lr={lr}, kl_beta={kl_beta}"
    )

    running_reward = 0.0
    running_acc = 0.0

    for step in range(1, steps + 1):
        progress = (step - 1) / max(1, steps)
        # Curriculum in RL: focus more on Level 2 (multi-hop) and Level 3 (memory reuse)
        r_lvl = rng.random()
        if progress < 0.30:
            lvl = 1 if r_lvl < 0.45 else (2 if r_lvl < 0.85 else 3)
        else:
            lvl = 1 if r_lvl < 0.15 else (2 if r_lvl < 0.55 else 3)

        ep = generate_episode(lvl, rng)

        # 1) Generate G interactive rollouts
        rollouts = [
            run_interactive_rollout(
                model=model,
                tokenizer=tokenizer,
                ep=ep,
                device=device,
                max_new_tokens=max_new_tokens,
                max_seq_len=seq_len,
                temperature=0.8,
                top_p=0.9,
                greedy=False,
            )
            for _ in range(group_size)
        ]

        rewards = [ro.reward for ro in rollouts]
        mean_r = sum(rewards) / len(rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
        acc_g = sum(1.0 for ro in rollouts if ro.correct) / len(rollouts)

        running_reward = 0.9 * running_reward + 0.1 * mean_r if step > 1 else mean_r
        running_acc = 0.9 * running_acc + 0.1 * acc_g if step > 1 else acc_g

        # Skip gradient update if all rollouts in the group received identical reward
        if std_r < 1e-6:
            if step % log_every == 0 or step == 1 or step == steps:
                print(
                    f"  [GRPO] step {step:4d}/{steps} | L{lvl} | reward {mean_r:.3f} (ema {running_reward:.3f}) "
                    f"| acc {acc_g:.2f} (ema {running_acc:.2f}) | skip(zero-var)"
                )
            continue

        advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        # 2) OOM-safe per-rollout gradient accumulation (peak VRAM = batch_size 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_pg_loss = 0.0
        total_kl = 0.0

        for ro, adv in zip(rollouts, advantages):
            if sum(ro.action_mask[1:]) == 0:
                continue
            pol_lp, mask = sequence_token_logprobs(model, ro.full_ids, ro.action_mask, device, amp_dtype, use_amp)
            with torch.no_grad():
                ref_lp, _ = sequence_token_logprobs(ref_model, ro.full_ids, ro.action_mask, device, amp_dtype, use_amp)

            denom = mask.sum().clamp_min(1.0)
            # Per-token KL approximation: exp(ref - pol) - (ref - pol) - 1
            log_ratio = (ref_lp - pol_lp).clamp(-10.0, 10.0)
            kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0
            kl_loss = (kl_per_tok * mask).sum() / denom

            pg_loss = -adv * (pol_lp * mask).sum() / denom
            loss = (pg_loss + kl_beta * kl_loss) / group_size
            loss.backward()

            total_pg_loss += float(pg_loss.item()) / group_size
            total_kl += float(kl_loss.item()) / group_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == "cuda" and step % 10 == 0:
            torch.cuda.empty_cache()

        if step % log_every == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            print(
                f"  [GRPO] step {step:4d}/{steps} | L{lvl} | reward {mean_r:.3f} (ema {running_reward:.3f}) "
                f"| acc {acc_g:.2f} (ema {running_acc:.2f}) | kl {total_kl:.4f} | {elapsed:.1f}s"
            )

    ckpt_path = output_dir / "stage2_grpo_latest.pt"
    torch.save({"stage": "grpo", "steps": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"[Stage 2: GRPO] Saved checkpoint -> {ckpt_path}")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 7. Evaluation across Level 1, 2, 3
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    tokenizer: CurriculumTokenizer,
    device: torch.device,
    eval_per_level: int = 10,
    max_new_tokens: int = 160,
    seq_len: int = 384,
    seed: int = 9999,
    show_samples: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    summary: dict[str, Any] = {}

    for lvl in (1, 2, 3):
        correct_cnt = 0
        format_cnt = 0
        mem_reuse_cnt = 0
        total_reward = 0.0
        sample_printed = False

        for _ in range(eval_per_level):
            ep = generate_episode(lvl, rng)
            ro = run_interactive_rollout(
                model=model,
                tokenizer=tokenizer,
                ep=ep,
                device=device,
                max_new_tokens=max_new_tokens,
                max_seq_len=seq_len,
                greedy=True,
            )
            correct_cnt += int(ro.correct)
            format_cnt += int(ro.format_ok)
            total_reward += ro.reward
            if lvl == 3 and ro.correct and len(ro.searches_made) == 0:
                mem_reuse_cnt += 1

            if show_samples and not sample_printed:
                sample_printed = True
                print(f"  [Sample L{lvl}] prompt: {ep.prompt.strip().splitlines()[-1]}")
                print(f"             expected={ep.expected_answer} | correct={ro.correct} | searches={ro.searches_made}")
                print(f"             output: {ro.completion_text[:220]}")

        summary[f"level_{lvl}"] = {
            "accuracy": correct_cnt / max(1, eval_per_level),
            "format_rate": format_cnt / max(1, eval_per_level),
            "avg_reward": total_reward / max(1, eval_per_level),
            **({"memory_reuse_rate": mem_reuse_cnt / max(1, eval_per_level)} if lvl == 3 else {}),
        }

    return summary


# ---------------------------------------------------------------------------
# 8. CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged Curriculum Learning (SFT -> GRPO) for Small Titans (MAC).")
    p.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    p.add_argument("--model_type", type=str, default="titans", choices=["titans", "tiny_transformer"])
    p.add_argument("--stage", type=str, default="all", choices=["sft", "grpo", "all", "eval"])
    p.add_argument("--sft_steps", type=int, default=600)
    p.add_argument("--grpo_steps", type=int, default=120)
    p.add_argument("--group_size", type=int, default=4, help="GRPO rollouts per prompt (4 is safe for 16GB VRAM)")
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--kl_beta", type=float, default=0.04)
    p.add_argument("--eval_per_level", type=int, default=10)
    p.add_argument("--sft_batch_size", type=int, default=0)
    p.add_argument("--sft_grad_accum", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="checkpoints/exp1_curriculum_rl")
    p.add_argument("--load_ckpt", type=str, default="", help="Optional .pt checkpoint to load before running")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    preset = PRESETS[args.preset]
    sft_batch = args.sft_batch_size or preset.sft_batch_size
    sft_accum = args.sft_grad_accum or preset.sft_grad_accum
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, amp_dtype = get_device_and_dtype()
    print(f"[Env] device={device}, amp_dtype={amp_dtype}, model_type={args.model_type}, preset={args.preset}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    vocab_path = output_dir / "vocab.json"
    if vocab_path.exists():
        tokenizer = CurriculumTokenizer.load(vocab_path)
    else:
        tokenizer = CurriculumTokenizer.build_default()
        tokenizer.save(vocab_path)
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size} (saved at {vocab_path})")

    model_cfg = {
        "num_tokens": tokenizer.vocab_size,
        "dim": preset.dim,
        "depth": preset.depth,
        "segment_len": preset.segment_len,
        "num_persist_mem_tokens": preset.num_persist_mem_tokens,
        "num_longterm_mem_tokens": preset.num_longterm_mem_tokens,
        "seq_len": preset.seq_len,
    }
    model = build_model(args.model_type, model_cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {args.model_type} params: {total_params / 1e6:.2f}M")

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] Loaded weights from {args.load_ckpt}")

    metrics: dict[str, Any] = {"preset": args.preset, "model_type": args.model_type, "preset_cfg": asdict(preset)}

    if args.stage in ("sft", "all"):
        run_sft_stage(
            model=model,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            steps=args.sft_steps,
            batch_size=sft_batch,
            grad_accum=sft_accum,
            seq_len=preset.seq_len,
            max_lr=preset.sft_lr,
            output_dir=output_dir,
            log_every=args.log_every,
            seed=args.seed,
        )
        print("\n[Eval after Stage 1: SFT]")
        metrics["after_sft"] = evaluate_model(
            model, tokenizer, device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_sft"], ensure_ascii=False, indent=2))

    if args.stage in ("grpo", "all"):
        run_grpo_stage(
            model=model,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            steps=args.grpo_steps,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
            lr=preset.grpo_lr,
            kl_beta=args.kl_beta,
            output_dir=output_dir,
            log_every=max(1, min(args.log_every, 10)),
            seed=args.seed + 1000,
        )
        print("\n[Eval after Stage 2: GRPO]")
        metrics["after_grpo"] = evaluate_model(
            model, tokenizer, device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_grpo"], ensure_ascii=False, indent=2))

    if args.stage == "eval":
        print("\n[Eval]")
        metrics["eval"] = evaluate_model(
            model, tokenizer, device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["eval"], ensure_ascii=False, indent=2))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
