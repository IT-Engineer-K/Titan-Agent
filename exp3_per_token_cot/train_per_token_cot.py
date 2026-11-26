"""
Experiment 3: Per-Token CoT (Stream vs Ephemeral) vs Turn-Level CoT on Titans (MAC)

Compares three Chain-of-Thought architectures on a semantic-token, multi-turn
memory + multi-hop compositional reasoning task:

  1. turn_cot (Baseline):
       One <think>...</think> block per turn before outputting 5 answer tokens
       <ans> y1 y2 y3 y4 y5 </ans>. Full <think> stays in the persistent stream.
  2. stream_token_cot (Per-Token CoT, Persistent / Stream):
       Before EACH output token y_k, emits a short <t> r_k </t> thought.
       All <t>...</t> blocks remain in the persistent Titans MAC stream across
       subsequent tokens and turns.
  3. ephemeral_token_cot (Per-Token CoT, Ephemeral / Quiet-STaR-like):
       Before EACH output token y_k, emits a short <t> r_k </t> thought, then
       EVICTS <t> r_k </t> from the persistent stream once y_k is chosen.
       Only clean output tokens <ans> y1..y5 </ans> accumulate in Titans MAC.

Task (Multi-Turn Memory + 5-Hop Compositional Rule Reasoning):
  - Premise slots (5): 時間, 予算, 優先, 経験, 目標 (semantic tokens)
  - Multi-turn 1-stream conversation:
      Initial premises -> distractors -> premise update -> Turn 1 consultation
      -> distractors -> premise update -> Turn 2 consultation
      -> distractors -> premise update -> Turn 3 consultation
  - Each consultation requires predicting 5 dependent semantic tokens:
      y1 (負荷)  <- f(時間, 優先)
      y2 (ツール) <- f(予算, 経験)
      y3 (ペース) <- f(y1, 時間, 優先)
      y4 (検証)  <- f(y2, 目標, 経験)
      y5 (戦略)  <- f(y1, y3, y4, 優先)

Designed for Kaggle T4 (16GB) OOM-safe execution (~25-35 min for all 3 modes)
and fast local CPU smoke testing (--preset micro --model_type tiny_transformer).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Semantic-Unit Vocabulary & Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|unk|>",
    "<think>",
    "</think>",
    "<t>",
    "</t>",
    "<ans>",
    "</ans>",
    "<turn>",
    "</turn>",
]

SLOT_NAMES = ["時間", "予算", "優先", "経験", "目標"]

SLOT_VALUES: dict[str, list[str]] = {
    "時間": ["極短", "短", "中", "週末"],
    "予算": ["無料", "小額", "十分"],
    "優先": ["継続", "速効", "品質", "低負荷"],
    "経験": ["初級", "中級", "上級"],
    "目標": ["副収入", "転職", "資格", "制作", "習慣"],
}

TOPICS = ["アプリ", "英語", "ブログ", "デザイン", "自動化", "資格学習", "発信", "研究"]

ANSWER_VALUES: dict[str, list[str]] = {
    "負荷": ["負荷:軽", "負荷:中", "負荷:重"],
    "ツール": ["ツール:無料", "ツール:AI課金", "ツール:教材購入"],
    "ペース": ["ペース:毎日小刻み", "ペース:平日固定", "ペース:週末集中", "ペース:短期スプリント"],
    "検証": ["検証:自己チェック", "検証:公開テスト", "検証:定量計測", "検証:レビュー依頼"],
    "戦略": [
        "戦略:最小継続型",
        "戦略:一点突破型",
        "戦略:設計重視型",
        "戦略:低コスト検証型",
        "戦略:週末集中型",
        "戦略:高速反復型",
    ],
}

ANSWER_SLOTS = ["負荷", "ツール", "ペース", "検証", "戦略"]

DISTRACTOR_EVENTS = [
    ["メモ:", "進捗", "確認中"],
    ["メモ:", "少し", "休憩"],
    ["メモ:", "別件", "整理中"],
    ["メモ:", "課題", "見直し"],
    ["メモ:", "環境", "調整済"],
    ["メモ:", "優先度", "検討中"],
    ["メモ:", "資料", "調査中"],
    ["メモ:", "記録", "保存済"],
]

STRUCTURAL_TOKENS = [
    "設定:",
    "更新:",
    "メモ:",
    "相談:",
    "記憶",
    "現在",
    "=>",
    "|",
    "進捗",
    "確認中",
    "少し",
    "休憩",
    "別件",
    "整理中",
    "課題",
    "見直し",
    "環境",
    "調整済",
    "優先度",
    "検討中",
    "資料",
    "調査中",
    "記録",
    "保存済",
]


class SemanticTokenizer:
    """Whitespace-separated semantic-unit tokenizer (1 concept = 1 token)."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<|pad|>"]
        self.eos_token_id = self.token_to_id["<|eos|>"]
        self.unk_token_id = self.token_to_id["<|unk|>"]
        self.t_start_id = self.token_to_id["<t>"]
        self.t_end_id = self.token_to_id["</t>"]
        self.ans_start_id = self.token_to_id["<ans>"]
        self.ans_end_id = self.token_to_id["</ans>"]
        self.vocab_size = len(self.token_to_id)

    @classmethod
    def build_default(cls) -> "SemanticTokenizer":
        vocab: dict[str, int] = {}

        def _add(tok: str) -> None:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        for tok in SPECIAL_TOKENS:
            _add(tok)
        for tok in STRUCTURAL_TOKENS:
            _add(tok)
        for tok in TOPICS:
            _add(tok)
        for slot in SLOT_NAMES:
            _add(slot)
            for val in SLOT_VALUES[slot]:
                _add(f"{slot}={val}")
        for ans_slot in ANSWER_SLOTS:
            for val in ANSWER_VALUES[ans_slot]:
                _add(val)

        return cls(vocab)

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def decode_tokens(self, ids: list[int], skip_pad: bool = True) -> list[str]:
        out: list[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "<|unk|>")
            if skip_pad and tok == "<|pad|>":
                continue
            out.append(tok)
        return out

    def decode_str(self, ids: list[int], skip_pad: bool = True) -> str:
        return " ".join(self.decode_tokens(ids, skip_pad=skip_pad))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in data.items()})


# ---------------------------------------------------------------------------
# 2. Multi-Hop Compositional Reasoning Rules & Multi-Turn Stream Generator
# ---------------------------------------------------------------------------

def solve_rules(slots: dict[str, str]) -> tuple[list[str], list[list[str]]]:
    """
    Deterministic 5-hop compositional reasoning over premise slots.
    Returns:
      - answers: [y1, y2, y3, y4, y5] (5 semantic tokens)
      - thoughts: [r1, r2, r3, r4, r5] (short semantic token list per hop)
    """
    t_val = slots["時間"]
    b_val = slots["予算"]
    p_val = slots["優先"]
    e_val = slots["経験"]
    g_val = slots["目標"]

    # Hop 1: 負荷 <- f(時間, 優先)
    if t_val == "極短" or p_val == "低負荷":
        y1 = "負荷:軽"
    elif (p_val == "速効" and t_val in ("中", "週末")) or (p_val == "品質" and t_val == "週末"):
        y1 = "負荷:重"
    else:
        y1 = "負荷:中"
    r1 = ["記憶", f"時間={t_val}", f"優先={p_val}", "=>", y1]

    # Hop 2: ツール <- f(予算, 経験)
    if b_val == "無料":
        y2 = "ツール:無料"
    elif (b_val == "十分" and e_val in ("中級", "上級")) or (b_val == "小額" and e_val == "上級"):
        y2 = "ツール:AI課金"
    else:
        y2 = "ツール:教材購入"
    r2 = ["記憶", f"予算={b_val}", f"経験={e_val}", "=>", y2]

    # Hop 3: ペース <- f(y1, 時間, 優先)
    if t_val == "週末":
        y3 = "ペース:週末集中"
    elif y1 == "負荷:重" and p_val == "速効":
        y3 = "ペース:短期スプリント"
    elif y1 == "負荷:軽" or p_val == "継続":
        y3 = "ペース:毎日小刻み"
    else:
        y3 = "ペース:平日固定"
    r3 = ["現在", y1, f"時間={t_val}", f"優先={p_val}", "=>", y3]

    # Hop 4: 検証 <- f(y2, 目標, 経験)
    if g_val in ("副収入", "制作") and e_val != "初級":
        y4 = "検証:公開テスト"
    elif g_val in ("転職", "資格") and y2 == "ツール:教材購入":
        y4 = "検証:定量計測"
    elif y2 == "ツール:AI課金":
        y4 = "検証:レビュー依頼"
    else:
        y4 = "検証:自己チェック"
    r4 = ["現在", y2, f"目標={g_val}", f"経験={e_val}", "=>", y4]

    # Hop 5: 戦略 <- f(y1, y3, y4, 優先)
    if y3 == "ペース:短期スプリント":
        y5 = "戦略:一点突破型"
    elif y3 == "ペース:週末集中":
        y5 = "戦略:週末集中型"
    elif y4 == "検証:公開テスト" and y1 != "負荷:軽":
        y5 = "戦略:高速反復型"
    elif p_val == "品質" and y4 in ("検証:定量計測", "検証:レビュー依頼"):
        y5 = "戦略:設計重視型"
    elif y1 == "負荷:軽":
        y5 = "戦略:最小継続型"
    else:
        y5 = "戦略:低コスト検証型"
    r5 = ["現在", y1, y3, y4, f"優先={p_val}", "=>", y5]

    return [y1, y2, y3, y4, y5], [r1, r2, r3, r4, r5]


@dataclass
class TurnSpec:
    turn_idx: int
    topic: str
    prefix_tokens: list[str]         # Events since previous turn ending with 相談: <topic>
    slots: dict[str, str]            # Ground-truth active slots at this turn
    stale_slots: dict[str, str]      # Updated slot -> old value before update
    answers: list[str]               # [y1, y2, y3, y4, y5]
    thoughts: list[list[str]]        # [r1, r2, r3, r4, r5]
    stale_answers: list[str] | None  # Answers if stale value were mistakenly used


@dataclass
class MultiTurnEpisode:
    turns: list[TurnSpec]
    num_distractors_per_turn: int


def _make_distractor_tokens(rng: random.Random, count: int) -> list[str]:
    toks: list[str] = []
    for _ in range(count):
        toks.extend(rng.choice(DISTRACTOR_EVENTS))
    return toks


def generate_multiturn_episode(
    rng: random.Random,
    num_turns: int = 3,
    num_distractors: int = 6,
    updates_per_turn: int = 1,
) -> MultiTurnEpisode:
    """
    Generates a continuous 1-stream multi-turn episode:
      - Initial premises for all 5 slots
      - For each turn: distractors -> premise update(s) -> distractors -> 相談: topic
    """
    slots = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
    topics = rng.sample(TOPICS, k=min(num_turns, len(TOPICS)))
    while len(topics) < num_turns:
        topics.append(rng.choice(TOPICS))

    turns: list[TurnSpec] = []
    for t_idx in range(num_turns):
        prefix: list[str] = []
        if t_idx == 0:
            prefix.append("設定:")
            for k in SLOT_NAMES:
                prefix.append(f"{k}={slots[k]}")

        d_before = num_distractors // 2
        d_after = num_distractors - d_before
        prefix.extend(_make_distractor_tokens(rng, d_before))

        stale_slots: dict[str, str] = {}
        if updates_per_turn > 0:
            upd_keys = rng.sample(SLOT_NAMES, k=min(updates_per_turn, len(SLOT_NAMES)))
            for k in upd_keys:
                old_val = slots[k]
                candidates = [v for v in SLOT_VALUES[k] if v != old_val]
                new_val = rng.choice(candidates)
                stale_slots[k] = old_val
                slots[k] = new_val
                prefix.extend(["更新:", f"{k}={new_val}"])

        prefix.extend(_make_distractor_tokens(rng, d_after))
        prefix.extend(["<turn>", "相談:", topics[t_idx]])

        answers, thoughts = solve_rules(slots)
        stale_answers = None
        if stale_slots:
            stale_copy = dict(slots)
            stale_copy.update(stale_slots)
            stale_answers, _ = solve_rules(stale_copy)

        turns.append(
            TurnSpec(
                turn_idx=t_idx,
                topic=topics[t_idx],
                prefix_tokens=prefix,
                slots=dict(slots),
                stale_slots=stale_slots,
                answers=answers,
                thoughts=thoughts,
                stale_answers=stale_answers,
            )
        )

    return MultiTurnEpisode(turns=turns, num_distractors_per_turn=num_distractors)


# ---------------------------------------------------------------------------
# 3. Mode-Specific Serialization (turn_cot / stream_token_cot / ephemeral_token_cot)
# ---------------------------------------------------------------------------

COT_MODES = ("turn_cot", "stream_token_cot", "ephemeral_token_cot")


def format_turn_completion_tokens(turn: TurnSpec, mode: str) -> list[str]:
    """
    Returns the persistent tokens appended to the stream for a completed turn.
    - turn_cot:            <think> r1 | r2 | r3 | r4 | r5 </think> <ans> y1..y5 </ans> </turn>
    - stream_token_cot:    <ans> <t> r1 </t> y1 ... <t> r5 </t> y5 </ans> </turn>
    - ephemeral_token_cot: <ans> y1 y2 y3 y4 y5 </ans> </turn>  (thoughts evicted!)
    """
    if mode == "turn_cot":
        toks = ["<think>"]
        for i, r in enumerate(turn.thoughts):
            if i > 0:
                toks.append("|")
            toks.extend(r)
        toks.append("</think>")
        toks.append("<ans>")
        toks.extend(turn.answers)
        toks.extend(["</ans>", "</turn>"])
        return toks

    if mode == "stream_token_cot":
        toks = ["<ans>"]
        for r, y in zip(turn.thoughts, turn.answers):
            toks.append("<t>")
            toks.extend(r)
            toks.append("</t>")
            toks.append(y)
        toks.extend(["</ans>", "</turn>"])
        return toks

    if mode == "ephemeral_token_cot":
        # Clean persistent history (no <t>...</t> retained)
        return ["<ans>", *turn.answers, "</ans>", "</turn>"]

    raise ValueError(f"Unknown mode: {mode}")


def build_sft_example(
    ep: MultiTurnEpisode,
    mode: str,
    tokenizer: SemanticTokenizer,
    max_seq_len: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """
    Builds one (input_ids, labels) training sequence for the given CoT mode.
    - turn_cot & stream_token_cot:
        Full multi-turn stream; loss is computed on all assistant turn completions.
    - ephemeral_token_cot:
        Samples a target turn and hop k in 0..4.
        Context = clean history of all previous turns + current prefix + <ans> + y_0..y_{k-1}
        Target  = <t> r_k </t> y_k [</ans> </turn> <|eos|> if k==4 else <|eos|>]
        Loss is computed ONLY on the ephemeral thought + output token!
    """
    if mode in ("turn_cot", "stream_token_cot"):
        seq_toks: list[str] = []
        mask: list[int] = []  # 1 = compute loss, 0 = ignore
        for t in ep.turns:
            seq_toks.extend(t.prefix_tokens)
            mask.extend([0] * len(t.prefix_tokens))
            comp = format_turn_completion_tokens(t, mode)
            seq_toks.extend(comp)
            mask.extend([1] * len(comp))
        seq_toks.append("<|eos|>")
        mask.append(1)

        ids = tokenizer.encode_tokens(seq_toks)
        labels = [tok_id if m == 1 else -100 for tok_id, m in zip(ids, mask)]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            labels = labels[:max_seq_len]
        return ids, labels

    if mode == "ephemeral_token_cot":
        target_t_idx = rng.randrange(len(ep.turns))
        target_hop = rng.randrange(len(ANSWER_SLOTS))

        prefix_toks: list[str] = []
        for idx in range(target_t_idx):
            t_prev = ep.turns[idx]
            prefix_toks.extend(t_prev.prefix_tokens)
            prefix_toks.extend(format_turn_completion_tokens(t_prev, "ephemeral_token_cot"))

        t_cur = ep.turns[target_t_idx]
        prefix_toks.extend(t_cur.prefix_tokens)
        prefix_toks.append("<ans>")
        prefix_toks.extend(t_cur.answers[:target_hop])

        target_toks = [
            "<t>",
            *t_cur.thoughts[target_hop],
            "</t>",
            t_cur.answers[target_hop],
            "<|eos|>",
        ]

        p_ids = tokenizer.encode_tokens(prefix_toks)
        t_ids = tokenizer.encode_tokens(target_toks)
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + list(t_ids)
        if len(ids) > max_seq_len:
            # Keep tail if over budget so target tokens are never truncated
            overflow = len(ids) - max_seq_len
            ids = ids[overflow:]
            labels = labels[overflow:]
        return ids, labels

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# 4. Models: Titans (MAC) & Fallback Causal Transformer
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    dim: int
    depth: int
    segment_len: int
    num_persist_mem_tokens: int
    num_longterm_mem_tokens: int
    seq_len: int
    batch_size: int
    grad_accum: int
    lr: float


PRESETS: dict[str, ModelPreset] = {
    # Fast local CPU/GPU smoke test (~1.4M params)
    "micro": ModelPreset(
        dim=128,
        depth=3,
        segment_len=16,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=8,
        seq_len=768,
        batch_size=4,
        grad_accum=2,
        lr=1.5e-3,
    ),
    # Kaggle T4/P100 16GB fast & OOM-safe (~10.1M params, segment_len=32 exercises 6-15 MAC segments)
    "tiny": ModelPreset(
        dim=256,
        depth=4,
        segment_len=32,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
        seq_len=768,
        batch_size=4,
        grad_accum=8,
        lr=8e-4,
    ),
    # Higher capacity (~22M params)
    "small": ModelPreset(
        dim=384,
        depth=6,
        segment_len=32,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        seq_len=768,
        batch_size=2,
        grad_accum=16,
        lr=5e-4,
    ),
}


class FallbackCausalTransformer(nn.Module):
    """Minimal causal Transformer (optionally sliding-window) for local tests / baselines."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        max_seq_len: int = 1024,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
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
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        if self.window_size > 0:
            too_far = torch.tril(
                torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool),
                diagonal=-self.window_size,
            )
            mask = mask | too_far
        h = self.blocks(h, mask=mask)
        return self.head(self.ln_f(h))


def build_model(model_type: str, cfg: dict[str, Any], device: torch.device) -> nn.Module:
    if model_type == "titans":
        from titans_pytorch import MemoryAsContextTransformer

        model = MemoryAsContextTransformer(
            num_tokens=cfg["num_tokens"],
            dim=cfg["dim"],
            depth=cfg["depth"],
            segment_len=cfg["segment_len"],
            num_persist_mem_tokens=cfg["num_persist_mem_tokens"],
            num_longterm_mem_tokens=cfg["num_longterm_mem_tokens"],
        )
        return model.to(device)

    window_size = cfg["segment_len"] if model_type == "sliding_transformer" else 0
    model = FallbackCausalTransformer(
        num_tokens=cfg["num_tokens"],
        dim=cfg["dim"],
        depth=cfg["depth"],
        max_seq_len=max(1024, cfg["seq_len"] + 128),
        window_size=window_size,
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
# 5. Training Loop (Fair Supervision Across Modes)
# ---------------------------------------------------------------------------

def collate_batch(
    items: list[tuple[list[int], list[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in items)
    bsz = len(items)
    x = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, lbls) in enumerate(items):
        n = len(ids)
        x[i, :n] = torch.tensor(ids, dtype=torch.long)
        y[i, :n] = torch.tensor(lbls, dtype=torch.long)
    return x, y


def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    p = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return max_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def sample_train_episode(rng: random.Random) -> MultiTurnEpisode:
    """Curriculum mix of 1-3 turns and 2-10 distractors per turn."""
    num_turns = rng.choices([1, 2, 3], weights=[0.25, 0.35, 0.40])[0]
    num_dist = rng.randint(2, 10)
    updates = rng.choices([0, 1], weights=[0.20, 0.80])[0]
    return generate_multiturn_episode(
        rng,
        num_turns=num_turns,
        num_distractors=num_dist,
        updates_per_turn=updates,
    )


def train_mode(
    mode: str,
    model_type: str,
    model_cfg: dict[str, Any],
    tokenizer: SemanticTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    output_dir: Path,
    log_every: int = 50,
    seed: int = 42,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = build_model(model_type, model_cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    use_amp = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_scaler)
    )
    warmup = max(1, min(40, steps // 10))

    model.train()
    t0 = time.time()
    print(
        f"\n[Train: {mode}] params={total_params / 1e6:.2f}M | steps={steps} "
        f"| batch={batch_size}x{grad_accum} | lr={lr}"
    )

    avg_seq_len_sum = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        cur_lr = cosine_lr(step - 1, warmup, steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        step_seq_lens = []

        for _ in range(grad_accum):
            batch_items = [
                build_sft_example(
                    sample_train_episode(rng),
                    mode=mode,
                    tokenizer=tokenizer,
                    max_seq_len=model_cfg["seq_len"],
                    rng=rng,
                )
                for _ in range(batch_size)
            ]
            step_seq_lens.extend(len(ids) for ids, _ in batch_items)
            x, y = collate_batch(batch_items, tokenizer.pad_token_id)
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

        avg_seq_len_sum += sum(step_seq_lens) / max(1, len(step_seq_lens))
        final_loss = accum_loss

        if step % log_every == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            print(
                f"  [{mode}] step {step:4d}/{steps} | loss {accum_loss:.4f} "
                f"| avg_len {sum(step_seq_lens)/len(step_seq_lens):.1f} | lr {cur_lr:.2e} | {elapsed:.1f}s"
            )

    train_secs = time.time() - t0
    ckpt_path = output_dir / f"{mode}_latest.pt"
    torch.save({"mode": mode, "steps": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_stats = {
        "steps": steps,
        "final_loss": round(final_loss, 4),
        "avg_train_seq_len": round(avg_seq_len_sum / max(1, steps), 1),
        "train_seconds": round(train_secs, 1),
    }
    return model, train_stats


# ---------------------------------------------------------------------------
# 6. Mode-Specific Greedy Inference & Multi-Turn Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _greedy_generate_until(
    model: nn.Module,
    prefix_ids: list[int],
    stop_ids: set[int],
    max_new_tokens: int,
    max_seq_len: int,
    device: torch.device,
) -> list[int]:
    seq = list(prefix_ids)
    if len(seq) >= max_seq_len - 4:
        seq = seq[-(max_seq_len - 4):]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if len(seq) >= max_seq_len:
            break
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        seq.append(next_id)
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return generated


def _extract_answers_from_tokens(tokens: list[str], mode: str) -> list[str]:
    """Extracts the predicted answer tokens inside <ans>...</ans>, ignoring <t>...</t>."""
    in_ans = False
    in_t = False
    preds: list[str] = []
    for tok in tokens:
        if tok == "<ans>":
            in_ans = True
            continue
        if tok == "</ans>":
            break
        if not in_ans:
            continue
        if tok == "<t>":
            in_t = True
            continue
        if tok == "</t>":
            in_t = False
            continue
        if not in_t:
            preds.append(tok)
    return preds[: len(ANSWER_SLOTS)]


@torch.no_grad()
def run_multiturn_inference(
    model: nn.Module,
    tokenizer: SemanticTokenizer,
    ep: MultiTurnEpisode,
    mode: str,
    device: torch.device,
    max_seq_len: int = 768,
) -> dict[str, Any]:
    """
    Runs a full multi-turn 1-stream rollout for the given mode.
    Returns per-turn predictions, persistent stream length, total generated tokens, and sample traces.
    """
    model.eval()
    stream_ids: list[int] = []
    turn_results: list[dict[str, Any]] = []
    total_generated_tokens = 0
    end_turn_id = tokenizer.token_to_id["</turn>"]
    eos_id = tokenizer.eos_token_id

    for t in ep.turns:
        stream_ids.extend(tokenizer.encode_tokens(t.prefix_tokens))
        prompt_len_at_turn = len(stream_ids)

        if mode in ("turn_cot", "stream_token_cot"):
            max_gen = 80 if mode == "turn_cot" else 90
            gen_ids = _greedy_generate_until(
                model=model,
                prefix_ids=stream_ids,
                stop_ids={end_turn_id, eos_id},
                max_new_tokens=max_gen,
                max_seq_len=max_seq_len,
                device=device,
            )
            total_generated_tokens += len(gen_ids)
            stream_ids.extend(gen_ids)
            gen_toks = tokenizer.decode_tokens(gen_ids)
            pred_answers = _extract_answers_from_tokens(gen_toks, mode)
            trace_str = " ".join(gen_toks)

        elif mode == "ephemeral_token_cot":
            # Append <ans> to persistent stream, then predict 5 tokens with ephemeral <t>...</t>
            stream_ids.append(tokenizer.ans_start_id)
            pred_answers = []
            hop_traces: list[str] = []

            for _hop in range(len(ANSWER_SLOTS)):
                gen_ids = _greedy_generate_until(
                    model=model,
                    prefix_ids=stream_ids,
                    stop_ids={eos_id},
                    max_new_tokens=20,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                total_generated_tokens += len(gen_ids)
                gen_toks = tokenizer.decode_tokens(gen_ids)
                hop_traces.append(" ".join(gen_toks))

                # Extract the single chosen answer token right after </t>
                chosen_tok = "<|unk|>"
                if "</t>" in gen_toks:
                    idx_after = gen_toks.index("</t>") + 1
                    if idx_after < len(gen_toks) and gen_toks[idx_after] != "<|eos|>":
                        chosen_tok = gen_toks[idx_after]
                elif gen_toks:
                    chosen_tok = gen_toks[0]

                pred_answers.append(chosen_tok)
                # EVICT <t>...</t>: append ONLY the clean answer token to persistent stream!
                stream_ids.append(tokenizer.token_to_id.get(chosen_tok, tokenizer.unk_token_id))

            # Close clean turn in persistent stream
            stream_ids.extend([tokenizer.ans_end_id, end_turn_id])
            trace_str = " || ".join(hop_traces)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Score this turn
        gt = t.answers
        padded_pred = pred_answers + [""] * max(0, len(gt) - len(pred_answers))
        hop_hits = [int(padded_pred[i] == gt[i]) for i in range(len(gt))]
        exact = int(all(hop_hits))
        stale_hit = 0
        if t.stale_answers is not None:
            # Check if any slot where stale != latest matched the stale answer
            for i in range(len(gt)):
                if t.stale_answers[i] != gt[i] and padded_pred[i] == t.stale_answers[i]:
                    stale_hit = 1
                    break

        turn_results.append(
            {
                "turn_idx": t.turn_idx,
                "prompt_len": prompt_len_at_turn,
                "stream_len_after": len(stream_ids),
                "exact": exact,
                "hop_hits": hop_hits,
                "stale_hit": stale_hit,
                "pred_answers": pred_answers,
                "gt_answers": gt,
                "trace": trace_str,
            }
        )

    return {
        "turns": turn_results,
        "final_stream_len": len(stream_ids),
        "total_generated_tokens": total_generated_tokens,
    }


def evaluate_suite(
    model: nn.Module,
    tokenizer: SemanticTokenizer,
    mode: str,
    device: torch.device,
    eval_episodes: int = 30,
    max_seq_len: int = 768,
    seed: int = 9999,
    show_sample: bool = True,
) -> dict[str, Any]:
    """
    Evaluates a trained model on 3 standardized benchmarks (identical episodes across modes):
      1. single_turn_d4:      1 turn, 4 distractors (baseline multi-hop reasoning)
      2. long_range_d16:      1 turn, 16 distractors (OOD long-range memory + reasoning)
      3. multi_turn_3t_d6:    3 consecutive turns in 1 stream, 6 distractors + update per turn
    """
    specs = {
        "single_turn_d4": dict(num_turns=1, num_distractors=4, updates_per_turn=1),
        "long_range_d16": dict(num_turns=1, num_distractors=16, updates_per_turn=1),
        "multi_turn_3t_d6": dict(num_turns=3, num_distractors=6, updates_per_turn=1),
    }

    results: dict[str, Any] = {}
    for bench_name, cfg in specs.items():
        rng = random.Random(seed + sum(ord(c) for c in bench_name))
        t0 = time.time()

        num_turns = cfg["num_turns"]
        turn_exact = [0] * num_turns
        turn_slot_sum = [0.0] * num_turns
        turn_y5 = [0] * num_turns
        turn_stale = [0] * num_turns
        turn_prompt_len = [0.0] * num_turns
        final_stream_len_sum = 0.0
        total_gen_sum = 0.0
        sample_printed = False

        for _ in range(eval_episodes):
            ep = generate_multiturn_episode(
                rng,
                num_turns=cfg["num_turns"],
                num_distractors=cfg["num_distractors"],
                updates_per_turn=cfg["updates_per_turn"],
            )
            out = run_multiturn_inference(
                model=model,
                tokenizer=tokenizer,
                ep=ep,
                mode=mode,
                device=device,
                max_seq_len=max_seq_len,
            )
            final_stream_len_sum += out["final_stream_len"]
            total_gen_sum += out["total_generated_tokens"]

            for tr in out["turns"]:
                ti = tr["turn_idx"]
                turn_exact[ti] += tr["exact"]
                turn_slot_sum[ti] += sum(tr["hop_hits"]) / len(ANSWER_SLOTS)
                turn_y5[ti] += tr["hop_hits"][-1]
                turn_stale[ti] += tr["stale_hit"]
                turn_prompt_len[ti] += tr["prompt_len"]

            if show_sample and not sample_printed and bench_name == "multi_turn_3t_d6":
                sample_printed = True
                last_tr = out["turns"][-1]
                print(
                    f"  [Sample {mode} | {bench_name} Turn 3] "
                    f"prompt_len={last_tr['prompt_len']} | exact={bool(last_tr['exact'])}"
                )
                print(f"    GT:    {last_tr['gt_answers']}")
                print(f"    Pred:  {last_tr['pred_answers']}")
                print(f"    Trace: {last_tr['trace'][:220]}")

        n = max(1, eval_episodes)
        per_turn = {}
        for ti in range(num_turns):
            per_turn[f"turn_{ti + 1}"] = {
                "exact_acc": round(turn_exact[ti] / n, 3),
                "slot_acc": round(turn_slot_sum[ti] / n, 3),
                "y5_strategy_acc": round(turn_y5[ti] / n, 3),
                "stale_rate": round(turn_stale[ti] / n, 3),
                "avg_prompt_len": round(turn_prompt_len[ti] / n, 1),
            }

        results[bench_name] = {
            "overall_exact_acc": round(sum(turn_exact) / (n * num_turns), 3),
            "overall_slot_acc": round(sum(turn_slot_sum) / (n * num_turns), 3),
            "overall_y5_acc": round(sum(turn_y5) / (n * num_turns), 3),
            "overall_stale_rate": round(sum(turn_stale) / (n * num_turns), 3),
            "avg_final_stream_len": round(final_stream_len_sum / n, 1),
            "avg_total_gen_tokens": round(total_gen_sum / n, 1),
            "eval_seconds": round(time.time() - t0, 1),
            "per_turn": per_turn,
        }

    return results


# ---------------------------------------------------------------------------
# 7. CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 3: Per-Token CoT (Stream vs Ephemeral) vs Turn-Level CoT on Titans (MAC)."
    )
    p.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    p.add_argument(
        "--model_type",
        type=str,
        default="titans",
        choices=["titans", "tiny_transformer", "sliding_transformer"],
    )
    p.add_argument(
        "--modes",
        type=str,
        default="turn_cot,stream_token_cot,ephemeral_token_cot",
        help="Comma-separated list of modes to run.",
    )
    p.add_argument("--steps", type=int, default=350, help="SFT steps for turn_cot & stream_token_cot.")
    p.add_argument(
        "--ephemeral_step_mult",
        type=float,
        default=2.0,
        help="Step multiplier for ephemeral_token_cot (each sequence trains 1 hop instead of all 5).",
    )
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="checkpoints/exp3_per_token_cot")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    batch_size = args.batch_size or preset.batch_size
    grad_accum = args.grad_accum or preset.grad_accum
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, amp_dtype = get_device_and_dtype()
    print(f"[Env] device={device}, amp_dtype={amp_dtype}, model_type={args.model_type}, preset={args.preset}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    tokenizer = SemanticTokenizer.build_default()
    vocab_path = output_dir / "vocab.json"
    tokenizer.save(vocab_path)
    print(f"[Tokenizer] semantic vocab_size={tokenizer.vocab_size} (saved at {vocab_path})")

    model_cfg = {
        "num_tokens": tokenizer.vocab_size,
        "dim": preset.dim,
        "depth": preset.depth,
        "segment_len": preset.segment_len,
        "num_persist_mem_tokens": preset.num_persist_mem_tokens,
        "num_longterm_mem_tokens": preset.num_longterm_mem_tokens,
        "seq_len": preset.seq_len,
    }

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in COT_MODES:
            raise ValueError(f"Invalid mode '{m}'. Choose from {COT_MODES}")

    all_metrics: dict[str, Any] = {
        "preset": args.preset,
        "model_type": args.model_type,
        "model_cfg": model_cfg,
        "modes": {},
    }

    for mode in modes:
        mode_steps = (
            max(1, int(round(args.steps * args.ephemeral_step_mult)))
            if mode == "ephemeral_token_cot"
            else args.steps
        )
        model, train_stats = train_mode(
            mode=mode,
            model_type=args.model_type,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            steps=mode_steps,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr=preset.lr,
            output_dir=output_dir,
            log_every=args.log_every,
            seed=args.seed,
        )
        eval_stats = evaluate_suite(
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            device=device,
            eval_episodes=args.eval_episodes,
            max_seq_len=preset.seq_len,
            seed=args.seed + 9999,
            show_sample=True,
        )
        all_metrics["modes"][mode] = {
            "train": train_stats,
            "eval": eval_stats,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Experiment 3 Comparison Summary ===")
    for mode, data in all_metrics["modes"].items():
        ev = data["eval"]
        print(
            f"  {mode:20s} | d4={ev['single_turn_d4']['overall_exact_acc']:.3f} "
            f"| d16={ev['long_range_d16']['overall_exact_acc']:.3f} "
            f"| 3T={ev['multi_turn_3t_d6']['overall_exact_acc']:.3f} "
            f"(T3={ev['multi_turn_3t_d6']['per_turn']['turn_3']['exact_acc']:.3f}) "
            f"| stream_len={ev['multi_turn_3t_d6']['avg_final_stream_len']:.1f}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
