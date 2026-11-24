"""
Experiment 2: Thinking Partner & Clarification Loop (SFT -> GRPO) for Small Titans (MAC).

Implements a 3-level "Thinking Partner (壁打ち・思考整理)" curriculum designed for
Kaggle free-tier GPUs (T4/P100 16GB) without OOM:
  - Level 1 (Clarification / 質問力):
      Spot 1 missing premise (目的 / 制約 / 優先軸) in an ambiguous consultation,
      ask a targeted clarifying question via <ask>...</ask>, receive <user>...</user>,
      and produce a structured <summary>...</summary> with an actionable policy.
  - Level 2 (Multi-turn Deepening / 深掘り＆論点整理):
      Spot 2 missing premises, ask sequentially across multiple turns without
      redundant questions, and synthesize all premises into a structured <summary>.
  - Level 3 (Long-Term Memory Wall-Bouncing / 長期記憶を活かした壁打ち):
      An earlier consultation in the same conversation stream already revealed one
      of the user's recurring premises (制約 or 優先軸), followed by distractor turns.
      In the new consultation, the model should recall that premise from memory
      WITHOUT re-asking it, ask only for the truly missing premise, and summarize.

Improvements over Experiment 1.5 based on T4 run diagnostics:
  - Shaped partial rewards (slot accuracy, ask precision, redundant-ask penalty) so
    GRPO groups maintain healthy reward variance instead of frequent skip(zero-var).
  - Higher GRPO focus on Level 2 & Level 3 with configurable sampling temperature.
  - Tracks GRPO update rate (non-skipped steps) and richer evaluation metrics
    (full accuracy, slot accuracy, ask precision, memory reuse rate).
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
# 1. Special Tags, Domain Vocabulary & Self-Contained Character Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|unk|>",
    "<think>",
    "</think>",
    "<ask>",
    "</ask>",
    "<user>",
    "</user>",
    "<summary>",
    "</summary>",
]

TOPICS = [
    "副業", "アプリ開発", "英語学習", "転職",
    "ブログ執筆", "資格勉強", "運動習慣", "読書習慣",
    "個人制作", "研究テーマ", "時間管理", "発信活動",
    "資産形成", "デザイン学習", "自動化ツール", "キャリア設計",
]

SLOT_NAMES = ["目的", "制約", "優先軸"]

SLOT_VALUES: dict[str, list[str]] = {
    "目的": [
        "収益化", "スキル習得", "習慣化", "実績作り",
        "キャリア転換", "好奇心探究", "業務効率化", "自己表現",
    ],
    "制約": [
        "平日30分", "週末のみ", "予算ゼロ", "1日1時間",
        "期限1か月", "スマホ中心", "低負荷", "夜間のみ",
    ],
    "優先軸": [
        "継続しやすさ", "短期成果", "楽しさ", "質重視",
        "低リスク", "独学完結",
    ],
}

# Deterministic policy mapping from 優先軸 (and 制約) so <summary> requires synthesis, not just copying
PRIORITY_TO_PLAN: dict[str, str] = {
    "継続しやすさ": "最小単位で毎日回す",
    "短期成果": "期限を切り一点集中",
    "楽しさ": "興味軸で小さく試す",
    "質重視": "設計を固めて深掘り",
    "低リスク": "無料枠で検証優先",
    "独学完結": "教材を絞り自走する",
}

CONSULT_TEMPLATES = [
    "相談: {topic}を考えている。{known_text}。どう進めるべき？",
    "相談: {topic}について思考を整理したい。{known_text}。方針を決めたい。",
    "相談: {topic}を始めたいが迷っている。{known_text}。",
]

DISTRACTOR_TURNS = [
    "メモ: 今の考えをそのまま保持中。",
    "メモ: 少し休憩してから次の相談へ移る。",
    "メモ: さっきの前提は今後も活かせそう。",
    "メモ: 別のテーマも続けて壁打ちしたい。",
    "メモ: 頭の中を少しずつ言語化している。",
]


class ThinkingTokenizer:
    """Deterministic special-tag + character tokenizer (zero external downloads)."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<|pad|>"]
        self.eos_token_id = self.token_to_id["<|eos|>"]
        self.unk_token_id = self.token_to_id["<|unk|>"]
        self.ask_end_id = self.token_to_id["</ask>"]
        self.vocab_size = len(self.token_to_id)

        tag_pattern = "|".join(re.escape(t) for t in SPECIAL_TOKENS)
        self._split_re = re.compile(f"({tag_pattern})")

    @classmethod
    def build_default(cls) -> "ThinkingTokenizer":
        chars: set[str] = set()
        for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-*/=|():.,!? \n、。":
            chars.add(ch)
        sample_texts = [
            "相談: テーマ: 既知: 記憶: 不足: 確認: 整理完了。全前提が揃った。",
            "方針=エラー: 該当なし",
            "テーマ=目的=制約=優先軸=方針=",
            "を聞く。次はが必要。前回の前提を再利用。",
            *TOPICS,
            *SLOT_NAMES,
            *[v for vals in SLOT_VALUES.values() for v in vals],
            *PRIORITY_TO_PLAN.values(),
            *CONSULT_TEMPLATES,
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
    def load(cls, path: Path) -> "ThinkingTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in data.items()})


# ---------------------------------------------------------------------------
# 2. Procedural 3-Level Thinking-Partner Generator & Mock User Environment
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    level: int
    prompt: str
    topic: str
    full_slots: dict[str, str]       # Ground truth for 目的, 制約, 優先軸
    known_slots: list[str]           # Already stated in the current consultation
    memory_slots: list[str]          # Stated in an earlier turn (Level 3)
    missing_slots: list[str]         # Must be asked via <ask>
    expected_plan: str
    ideal_completion: str


def _format_known(slots: dict[str, str], keys: list[str]) -> str:
    return "、".join(f"{k}は{slots[k]}" for k in keys)


def _format_kv(slots: dict[str, str], keys: list[str]) -> str:
    return "、".join(f"{k}={slots[k]}" for k in keys)


def _build_summary(topic: str, slots: dict[str, str], plan: str) -> str:
    return (
        f"テーマ={topic} | 目的={slots['目的']} | 制約={slots['制約']} "
        f"| 優先軸={slots['優先軸']} | 方針={plan}"
    )


def generate_episode(level: int, rng: random.Random, num_distractors: int = 1) -> Episode:
    """Generates a fresh wall-bouncing consultation episode with a simulated user."""
    assert level in (1, 2, 3), f"Invalid level: {level}"

    if level == 1:
        topic = rng.choice(TOPICS)
        full_slots = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        missing = rng.choice(SLOT_NAMES)
        known = [k for k in SLOT_NAMES if k != missing]
        plan = PRIORITY_TO_PLAN[full_slots["優先軸"]]

        tmpl = rng.choice(CONSULT_TEMPLATES)
        prompt = tmpl.format(topic=topic, known_text=_format_known(full_slots, known)) + "\n"
        summary_str = _build_summary(topic, full_slots, plan)
        ideal = (
            f"<think>テーマ: {topic}。既知: {_format_kv(full_slots, known)}。不足: {missing}</think>"
            f"<ask>{missing}</ask>"
            f"<user>{missing}: {full_slots[missing]}</user>"
            f"<think>整理完了。全前提が揃った。</think>"
            f"<summary>{summary_str}</summary><|eos|>"
        )
        return Episode(
            level=1,
            prompt=prompt,
            topic=topic,
            full_slots=full_slots,
            known_slots=known,
            memory_slots=[],
            missing_slots=[missing],
            expected_plan=plan,
            ideal_completion=ideal,
        )

    if level == 2:
        topic = rng.choice(TOPICS)
        full_slots = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        known_slot = rng.choice(SLOT_NAMES)
        missing = [k for k in SLOT_NAMES if k != known_slot]
        m1, m2 = missing[0], missing[1]
        plan = PRIORITY_TO_PLAN[full_slots["優先軸"]]

        tmpl = rng.choice(CONSULT_TEMPLATES)
        prompt = tmpl.format(topic=topic, known_text=_format_known(full_slots, [known_slot])) + "\n"
        summary_str = _build_summary(topic, full_slots, plan)
        ideal = (
            f"<think>テーマ: {topic}。既知: {known_slot}={full_slots[known_slot]}。不足: {m1}、{m2}</think>"
            f"<ask>{m1}</ask>"
            f"<user>{m1}: {full_slots[m1]}</user>"
            f"<think>確認: {m1}={full_slots[m1]}。次は{m2}が必要。不足: {m2}</think>"
            f"<ask>{m2}</ask>"
            f"<user>{m2}: {full_slots[m2]}</user>"
            f"<think>整理完了。全前提が揃った。</think>"
            f"<summary>{summary_str}</summary><|eos|>"
        )
        return Episode(
            level=2,
            prompt=prompt,
            topic=topic,
            full_slots=full_slots,
            known_slots=[known_slot],
            memory_slots=[],
            missing_slots=[m1, m2],
            expected_plan=plan,
            ideal_completion=ideal,
        )

    # Level 3: Long-term memory reuse across two consultations in one stream
    topic1, topic2 = rng.sample(TOPICS, 2)
    mem_slot = rng.choice(["制約", "優先軸"])
    shared_val = rng.choice(SLOT_VALUES[mem_slot])

    # Turn 1 consultation (already completed in history; establishes user's recurring premise)
    slots1 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
    slots1[mem_slot] = shared_val
    plan1 = PRIORITY_TO_PLAN[slots1["優先軸"]]
    t1_known = [k for k in SLOT_NAMES if k != mem_slot]
    t1_tmpl = rng.choice(CONSULT_TEMPLATES)
    t1_summary = _build_summary(topic1, slots1, plan1)
    turn1 = (
        t1_tmpl.format(topic=topic1, known_text=_format_known(slots1, t1_known)) + "\n"
        f"<think>テーマ: {topic1}。既知: {_format_kv(slots1, t1_known)}。不足: {mem_slot}</think>"
        f"<ask>{mem_slot}</ask>"
        f"<user>{mem_slot}: {shared_val}</user>"
        f"<think>整理完了。全前提が揃った。</think>"
        f"<summary>{t1_summary}</summary>\n"
    )

    distractors = "\n".join(rng.choice(DISTRACTOR_TURNS) for _ in range(max(1, num_distractors))) + "\n"

    # Turn 2 consultation: states 1 slot, reuses mem_slot from memory, misses 1 remaining slot
    slots2 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
    slots2[mem_slot] = shared_val
    remaining = [k for k in SLOT_NAMES if k != mem_slot]
    known2 = rng.choice(remaining)
    missing2 = [k for k in remaining if k != known2][0]
    plan2 = PRIORITY_TO_PLAN[slots2["優先軸"]]

    t2_tmpl = rng.choice(CONSULT_TEMPLATES)
    t2_prompt = t2_tmpl.format(topic=topic2, known_text=_format_known(slots2, [known2])) + "\n"
    prompt = turn1 + distractors + t2_prompt
    t2_summary = _build_summary(topic2, slots2, plan2)
    ideal = (
        f"<think>テーマ: {topic2}。既知: {known2}={slots2[known2]}。"
        f"記憶: {mem_slot}={shared_val}。不足: {missing2}</think>"
        f"<ask>{missing2}</ask>"
        f"<user>{missing2}: {slots2[missing2]}</user>"
        f"<think>整理完了。全前提が揃った。</think>"
        f"<summary>{t2_summary}</summary><|eos|>"
    )
    return Episode(
        level=3,
        prompt=prompt,
        topic=topic2,
        full_slots=slots2,
        known_slots=[known2],
        memory_slots=[mem_slot],
        missing_slots=[missing2],
        expected_plan=plan2,
        ideal_completion=ideal,
    )


def tokenize_sft_episode(
    ep: Episode,
    tokenizer: ThinkingTokenizer,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenizes an SFT episode and builds masked labels:
      - Prompt tokens -> -100
      - Environment <user>...</user> tokens -> -100
      - Model-generated tokens (<think>, <ask>, <summary>, <|eos|>) -> target token ID
    """
    prompt_ids = tokenizer.encode(ep.prompt)
    comp_ids = tokenizer.encode(ep.ideal_completion)

    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + list(comp_ids)

    user_start = tokenizer.token_to_id["<user>"]
    user_end = tokenizer.token_to_id["</user>"]
    in_user = False
    for idx in range(len(prompt_ids), len(labels)):
        tok_id = input_ids[idx]
        if tok_id == user_start:
            in_user = True
            labels[idx] = -100
        elif in_user:
            labels[idx] = -100
            if tok_id == user_end:
                in_user = False

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
        seq_len=448,
        sft_batch_size=8,
        sft_grad_accum=2,
        sft_lr=1.5e-3,
        grpo_lr=1e-4,
    ),
    # Kaggle T4/P100 fast curriculum run (~10M params, <4GB VRAM)
    "tiny": ModelPreset(
        dim=256,
        depth=4,
        segment_len=64,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
        seq_len=448,
        sft_batch_size=4,
        sft_grad_accum=8,
        sft_lr=8e-4,
        grpo_lr=5e-5,
    ),
    # Kaggle T4/P100 higher-capacity run (~22M params, OOM-safe with batch_size=2)
    "small": ModelPreset(
        dim=384,
        depth=6,
        segment_len=64,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        seq_len=448,
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
        max_seq_len=max(512, model_cfg.get("seq_len", 448) + 128),
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
# 4. Interactive Clarifying-Dialogue Rollout & Shaped Reward Function
# ---------------------------------------------------------------------------

ASK_RE = re.compile(r"<ask>([^<>]+)</ask>")
SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class RolloutResult:
    full_ids: list[int]
    action_mask: list[int]  # 1 = model-generated token, 0 = prompt or <user>
    completion_text: str
    asks_made: list[str]
    reward: float
    correct: bool
    slot_accuracy: float
    ask_exact: bool
    memory_reused: bool
    format_ok: bool


def _parse_summary(summary_body: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for seg in summary_body.split("|"):
        seg = seg.strip()
        if "=" in seg:
            k, v = seg.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def score_rollout(
    ep: Episode,
    completion_text: str,
    asks_made: list[str],
) -> tuple[float, bool, float, bool, bool, bool]:
    """
    Automatic verifier for thinking-partner rollouts with shaped partial rewards:
      - Format adherence (<think> and <summary>): +0.20
      - Targeted clarification (asking each missing slot once): up to +0.50
      - Penalties for redundant asks (already known / already asked / memory slot) & invalid slots
      - Summary slot accuracy across 5 fields (テーマ, 目的, 制約, 優先軸, 方針): up to +1.00
      - Level 3 memory reuse bonus (correct without re-asking memory slot): +0.30
    """
    has_think = bool(THINK_RE.search(completion_text))
    sum_match = SUMMARY_RE.search(completion_text)
    format_ok = has_think and (sum_match is not None)

    reward = 0.20 if format_ok else 0.0

    # 1) Evaluate clarifying questions
    missing_set = set(ep.missing_slots)
    known_or_mem = set(ep.known_slots) | set(ep.memory_slots)
    seen_asks: set[str] = set()
    hit_missing: set[str] = set()
    redundant_cnt = 0
    invalid_cnt = 0
    reasked_memory = False

    for q in asks_made:
        if q not in SLOT_NAMES:
            invalid_cnt += 1
            continue
        if q in seen_asks:
            redundant_cnt += 1
            continue
        seen_asks.add(q)
        if q in missing_set:
            hit_missing.add(q)
        elif q in known_or_mem:
            redundant_cnt += 1
            if q in ep.memory_slots:
                reasked_memory = True

    if missing_set:
        reward += 0.50 * (len(hit_missing) / len(missing_set))
    reward -= 0.25 * redundant_cnt
    reward -= 0.20 * invalid_cnt
    if len(asks_made) > 3:
        reward -= 0.20

    ask_exact = (hit_missing == missing_set) and (redundant_cnt == 0) and (invalid_cnt == 0)

    # 2) Evaluate structured summary
    expected_fields = {
        "テーマ": ep.topic,
        "目的": ep.full_slots["目的"],
        "制約": ep.full_slots["制約"],
        "優先軸": ep.full_slots["優先軸"],
        "方針": ep.expected_plan,
    }
    matched_fields = 0
    if sum_match is not None:
        parsed = _parse_summary(sum_match.group(1))
        for k, expected_v in expected_fields.items():
            if parsed.get(k) == expected_v:
                matched_fields += 1

    slot_accuracy = matched_fields / len(expected_fields)
    reward += 1.00 * slot_accuracy
    correct = (matched_fields == len(expected_fields)) and format_ok

    # 3) Level 3 memory reuse bonus
    memory_reused = bool(ep.memory_slots) and correct and (not reasked_memory)
    if memory_reused:
        reward += 0.30

    return reward, correct, slot_accuracy, ask_exact, memory_reused, format_ok


@torch.no_grad()
def run_interactive_rollout(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    ep: Episode,
    device: torch.device,
    max_new_tokens: int = 190,
    max_seq_len: int = 448,
    temperature: float = 0.95,
    top_p: float = 0.92,
    greedy: bool = False,
    max_asks: int = 3,
) -> RolloutResult:
    """
    Autoregressively generates a wall-bouncing response while intercepting <ask>...</ask>
    and injecting <user>...</user> from the episode's simulated user state.
    """
    model.eval()
    prompt_ids = tokenizer.encode(ep.prompt)
    seq = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    asks_made: list[str] = []

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

        # Intercept </ask> and inject simulated user's <user>...</user> reply
        if next_id == tokenizer.ask_end_id and len(asks_made) < max_asks:
            comp_so_far = tokenizer.decode(seq[len(prompt_ids):])
            matches = ASK_RE.findall(comp_so_far)
            if matches:
                slot_q = matches[-1].strip()
                asks_made.append(slot_q)
                if slot_q in ep.full_slots:
                    reply_str = f"{slot_q}: {ep.full_slots[slot_q]}"
                else:
                    reply_str = "エラー: 該当なし"
                env_ids = tokenizer.encode(f"<user>{reply_str}</user>")
                if len(seq) + len(env_ids) < max_seq_len:
                    seq.extend(env_ids)
                    action_mask.extend([0] * len(env_ids))

    completion_text = tokenizer.decode(seq[len(prompt_ids):])
    reward, correct, slot_acc, ask_exact, mem_reused, format_ok = score_rollout(
        ep, completion_text, asks_made
    )
    return RolloutResult(
        full_ids=seq,
        action_mask=action_mask,
        completion_text=completion_text,
        asks_made=asks_made,
        reward=reward,
        correct=correct,
        slot_accuracy=slot_acc,
        ask_exact=ask_exact,
        memory_reused=mem_reused,
        format_ok=format_ok,
    )


# ---------------------------------------------------------------------------
# 5. Stage 1: Curriculum SFT (Foundation Building)
# ---------------------------------------------------------------------------

def sample_sft_level(progress: float, rng: random.Random) -> int:
    r = rng.random()
    if progress < 0.35:
        return 1 if r < 0.75 else 2
    if progress < 0.70:
        return 1 if r < 0.25 else (2 if r < 0.75 else 3)
    return 1 if r < 0.15 else (2 if r < 0.55 else 3)


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
    tokenizer: ThinkingTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    max_lr: float,
    output_dir: Path,
    log_every: int = 25,
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
    warmup = max(1, min(60, steps // 10))

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
                    generate_episode(sample_sft_level(progress, rng), rng, num_distractors=1),
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
# 6. Stage 2: GRPO (Harder-Curriculum & Variance-Aware RL)
# ---------------------------------------------------------------------------

def sequence_token_logprobs(
    model: nn.Module,
    full_ids: list[int],
    action_mask: list[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    mask = torch.tensor([action_mask[1:]], dtype=torch.float32, device=device)

    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    with amp_ctx:
        logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chosen_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen_lp.squeeze(0), mask.squeeze(0)


def run_grpo_stage(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int,
    group_size: int,
    max_new_tokens: int,
    seq_len: int,
    lr: float,
    kl_beta: float,
    temperature: float,
    output_dir: Path,
    log_every: int = 10,
    seed: int = 1000,
) -> dict[str, Any]:
    """
    OOM-safe GRPO focused on Level 2 (multi-ask) & Level 3 (memory reuse with 1-2 distractors).
    Tracks how many steps had non-zero reward variance and performed a gradient update.
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
        f"max_new_tokens={max_new_tokens}, temp={temperature}, lr={lr}, kl_beta={kl_beta}"
    )

    running_reward = 0.0
    running_acc = 0.0
    updated_steps = 0

    for step in range(1, steps + 1):
        # Emphasize harder multi-turn (L2) and memory-reuse (L3) episodes during RL
        r_lvl = rng.random()
        if r_lvl < 0.10:
            lvl = 1
        elif r_lvl < 0.55:
            lvl = 2
        else:
            lvl = 3

        # Slightly harder distractors (1 or 2) in L3 during RL to challenge memory retention
        n_dist = rng.choice([1, 2]) if lvl == 3 else 1
        ep = generate_episode(lvl, rng, num_distractors=n_dist)

        rollouts = [
            run_interactive_rollout(
                model=model,
                tokenizer=tokenizer,
                ep=ep,
                device=device,
                max_new_tokens=max_new_tokens,
                max_seq_len=seq_len,
                temperature=temperature,
                top_p=0.92,
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

        if std_r < 1e-6:
            if step % log_every == 0 or step == 1 or step == steps:
                print(
                    f"  [GRPO] step {step:4d}/{steps} | L{lvl} | r {mean_r:.3f} (ema {running_reward:.3f}) "
                    f"| acc {acc_g:.2f} | upd {updated_steps}/{step} | skip(zero-var)"
                )
            continue

        updated_steps += 1
        advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_kl = 0.0

        for ro, adv in zip(rollouts, advantages):
            if sum(ro.action_mask[1:]) == 0:
                continue
            pol_lp, mask = sequence_token_logprobs(model, ro.full_ids, ro.action_mask, device, amp_dtype, use_amp)
            with torch.no_grad():
                ref_lp, _ = sequence_token_logprobs(ref_model, ro.full_ids, ro.action_mask, device, amp_dtype, use_amp)

            denom = mask.sum().clamp_min(1.0)
            log_ratio = (ref_lp - pol_lp).clamp(-10.0, 10.0)
            kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0
            kl_loss = (kl_per_tok * mask).sum() / denom

            pg_loss = -adv * (pol_lp * mask).sum() / denom
            loss = (pg_loss + kl_beta * kl_loss) / group_size
            loss.backward()
            total_kl += float(kl_loss.item()) / group_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == "cuda" and step % 10 == 0:
            torch.cuda.empty_cache()

        if step % log_every == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            print(
                f"  [GRPO] step {step:4d}/{steps} | L{lvl} | r {mean_r:.3f} (ema {running_reward:.3f}) "
                f"| acc {acc_g:.2f} | upd {updated_steps}/{step} | kl {total_kl:.4f} | {elapsed:.1f}s"
            )

    ckpt_path = output_dir / "stage2_grpo_latest.pt"
    torch.save({"stage": "grpo", "steps": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"[Stage 2: GRPO] Saved checkpoint -> {ckpt_path} (updated {updated_steps}/{steps} steps)")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"updated_steps": updated_steps, "total_steps": steps, "update_rate": round(updated_steps / max(1, steps), 3)}


# ---------------------------------------------------------------------------
# 7. Evaluation across Level 1, 2, 3
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    device: torch.device,
    eval_per_level: int = 40,
    max_new_tokens: int = 190,
    seq_len: int = 448,
    seed: int = 9999,
    show_samples: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    summary: dict[str, Any] = {}

    for lvl in (1, 2, 3):
        correct_cnt = 0
        format_cnt = 0
        ask_exact_cnt = 0
        mem_reuse_cnt = 0
        slot_acc_sum = 0.0
        total_reward = 0.0
        sample_printed = False

        for _ in range(eval_per_level):
            ep = generate_episode(lvl, rng, num_distractors=1)
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
            ask_exact_cnt += int(ro.ask_exact)
            mem_reuse_cnt += int(ro.memory_reused)
            slot_acc_sum += ro.slot_accuracy
            total_reward += ro.reward

            if show_samples and not sample_printed:
                sample_printed = True
                print(f"  [Sample L{lvl}] prompt: {ep.prompt.strip().splitlines()[-1]}")
                print(
                    f"             missing={ep.missing_slots} | mem={ep.memory_slots} "
                    f"| asks={ro.asks_made} | correct={ro.correct} | reward={ro.reward:.2f}"
                )
                print(f"             output: {ro.completion_text}")

        n = max(1, eval_per_level)
        summary[f"level_{lvl}"] = {
            "accuracy": round(correct_cnt / n, 3),
            "slot_accuracy": round(slot_acc_sum / n, 3),
            "ask_exact_rate": round(ask_exact_cnt / n, 3),
            "format_rate": round(format_cnt / n, 3),
            "avg_reward": round(total_reward / n, 3),
            **({"memory_reuse_rate": round(mem_reuse_cnt / n, 3)} if lvl == 3 else {}),
        }

    return summary


# ---------------------------------------------------------------------------
# 8. CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: Thinking Partner (SFT -> GRPO) for Small Titans (MAC).")
    p.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    p.add_argument("--model_type", type=str, default="titans", choices=["titans", "tiny_transformer"])
    p.add_argument("--stage", type=str, default="all", choices=["sft", "grpo", "all", "eval"])
    p.add_argument("--sft_steps", type=int, default=450)
    p.add_argument("--grpo_steps", type=int, default=120)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--grpo_temp", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=190)
    p.add_argument("--kl_beta", type=float, default=0.04)
    p.add_argument("--eval_per_level", type=int, default=40)
    p.add_argument("--sft_batch_size", type=int, default=0)
    p.add_argument("--sft_grad_accum", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="checkpoints/exp2_thinking_partner")
    p.add_argument("--load_ckpt", type=str, default="")
    p.add_argument("--log_every", type=int, default=25)
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
        tokenizer = ThinkingTokenizer.load(vocab_path)
    else:
        tokenizer = ThinkingTokenizer.build_default()
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
            model,
            tokenizer,
            device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_sft"], ensure_ascii=False, indent=2))

    if args.stage in ("grpo", "all"):
        metrics["grpo_stats"] = run_grpo_stage(
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
            temperature=args.grpo_temp,
            output_dir=output_dir,
            log_every=max(1, min(args.log_every, 10)),
            seed=args.seed + 1000,
        )
        print("\n[Eval after Stage 2: GRPO]")
        metrics["after_grpo"] = evaluate_model(
            model,
            tokenizer,
            device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_grpo"], ensure_ascii=False, indent=2))

    if args.stage == "eval":
        print("\n[Eval]")
        metrics["eval"] = evaluate_model(
            model,
            tokenizer,
            device,
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
