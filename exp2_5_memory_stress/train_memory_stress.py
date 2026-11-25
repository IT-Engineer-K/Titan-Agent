"""
Experiment 2.5: Long-Range Memory, Memory Overwrite & Composite Recall Stress Test
for Small Titans (MAC) (SFT -> GRPO).

Designed for Kaggle free-tier GPUs (T4/P100 16GB) without OOM, pushing past the
ceiling reached in Exp 2 (where SFT hit 100% on L1-L3 and GRPO mostly skipped):

  - Level 1 (Baseline Memory Reuse):
      Short-range memory reuse (1-2 distractors). 1 slot from Turn 1, 1 known, 1 missing.
  - Level 2 (Long-Range Memory Retention):
      Long-range memory reuse across many distractor turns (SFT: 4-8, GRPO/Eval: 10-16),
      crossing 10+ MAC segments (segment_len=64, seq_len up to 1024).
  - Level 3 (Memory Overwrite / Update):
      Turn 1 establishes a premise (制約 or 優先軸), then an explicit update turn later
      changes that premise (followed by more distractors). In Turn 2, the model must
      use the UPDATED value (not the stale Turn 1 value) without re-asking.
  - Level 4 (Composite Multi-Slot Recall + Overwrite + Long Range):
      Turn 1 establishes BOTH 制約 and 優先軸. Mid-stream, one of them is updated,
      surrounded by distractors. Turn 2 states only the topic (or 目的), requiring the
      model to recall BOTH premises (1 original + 1 updated) simultaneously and ask
      only for 目的 if missing.

Curriculum split so GRPO has genuine headroom:
  - Stage 1 (SFT): Focuses mainly on L1-L3 with moderate distractor distances (1-8)
    and light exposure to L4, stopping before full memorization.
  - Stage 2 (GRPO): Targets harder long-range L2 (10-16 distractors), L3 overwrite,
    and L4 composite stress with shaped rewards (stale-memory penalty, overwrite bonus,
    multi-memory recall bonus, strict format check).
  - Distance Curve Eval: Measures accuracy & memory reuse across distractor counts
    d in [2, 6, 12, 18] to visualize Titans MAC's retention curve.
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

CONSULT_NO_KNOWN_TEMPLATES = [
    "相談: {topic}を考えている。前の前提を踏まえて方針を決めたい。",
    "相談: {topic}について思考を整理したい。どう進めるべき？",
    "相談: {topic}を始めたいが迷っている。整理してほしい。",
]

UPDATE_TEMPLATES = [
    "更新: 状況が変わり、{slot}は{new_val}に変更した。",
    "更新: 方針を見直し、今後の{slot}は{new_val}にする。",
    "更新: 今の{slot}は{new_val}へ切り替えている。",
]

DISTRACTOR_TURNS = [
    "メモ: 今の考えをそのまま保持中。",
    "メモ: 少し休憩してから次の相談へ移る。",
    "メモ: 別のテーマも続けて壁打ちしたい。",
    "メモ: 頭の中を少しずつ言語化している。",
    "メモ: 今日はタスクの優先順位を見直した。",
    "メモ: 気になっている論点をメモに残しておく。",
    "メモ: 進め方のアイデアをいくつか比較している。",
    "メモ: 細かい条件は後でまとめて確認しよう。",
    "メモ: 作業環境を整えて集中しやすい状態にした。",
    "メモ: 過去のメモを読み返して方向性を確認中。",
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
            "相談: テーマ: 既知: 記憶: 更新記憶: 不足: 確認: 整理完了。全前提が揃った。",
            "方針=エラー: 該当なし",
            "テーマ=目的=制約=優先軸=方針=",
            "を聞く。次はが必要。前回の前提を再利用。最新値を使用。",
            *TOPICS,
            *SLOT_NAMES,
            *[v for vals in SLOT_VALUES.values() for v in vals],
            *PRIORITY_TO_PLAN.values(),
            *CONSULT_TEMPLATES,
            *CONSULT_NO_KNOWN_TEMPLATES,
            *UPDATE_TEMPLATES,
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
# 2. Procedural 4-Level Memory-Stress Episode Generator
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    level: int
    prompt: str
    topic: str
    full_slots: dict[str, str]       # Ground truth (latest values) for 目的, 制約, 優先軸
    known_slots: list[str]           # Stated in Turn 2 consultation
    memory_slots: list[str]          # Must be recalled from earlier turns (not re-asked)
    updated_slots: dict[str, str]    # slot -> stale (old) value before overwrite (L3/L4)
    missing_slots: list[str]         # Must be asked via <ask>
    expected_plan: str
    ideal_completion: str
    num_distractors: int


def _format_known(slots: dict[str, str], keys: list[str]) -> str:
    return "、".join(f"{k}は{slots[k]}" for k in keys)


def _format_kv(slots: dict[str, str], keys: list[str]) -> str:
    return "、".join(f"{k}={slots[k]}" for k in keys)


def _build_summary(topic: str, slots: dict[str, str], plan: str) -> str:
    return (
        f"テーマ={topic} | 目的={slots['目的']} | 制約={slots['制約']} "
        f"| 優先軸={slots['優先軸']} | 方針={plan}"
    )


def _make_distractors(rng: random.Random, count: int) -> str:
    if count <= 0:
        return ""
    return "\n".join(rng.choice(DISTRACTOR_TURNS) for _ in range(count)) + "\n"


def _build_turn1(
    rng: random.Random,
    topic1: str,
    slots1: dict[str, str],
    asked_slot: str,
) -> str:
    plan1 = PRIORITY_TO_PLAN[slots1["優先軸"]]
    t1_known = [k for k in SLOT_NAMES if k != asked_slot]
    t1_tmpl = rng.choice(CONSULT_TEMPLATES)
    t1_summary = _build_summary(topic1, slots1, plan1)
    return (
        t1_tmpl.format(topic=topic1, known_text=_format_known(slots1, t1_known)) + "\n"
        f"<think>テーマ: {topic1}。既知: {_format_kv(slots1, t1_known)}。不足: {asked_slot}</think>"
        f"<ask>{asked_slot}</ask>"
        f"<user>{asked_slot}: {slots1[asked_slot]}</user>"
        f"<think>整理完了。全前提が揃った。</think>"
        f"<summary>{t1_summary}</summary>\n"
    )


def generate_episode(
    level: int,
    rng: random.Random,
    num_distractors: int | None = None,
) -> Episode:
    """
    Generates a memory-stress wall-bouncing episode:
      - Level 1: Short-range single-slot memory reuse (default 1-2 distractors)
      - Level 2: Long-range single-slot memory reuse (default 6-12 distractors)
      - Level 3: Memory overwrite / update (Turn 1 -> distractors -> Update -> distractors -> Turn 2)
      - Level 4: Composite 2-slot memory recall (制約 + 優先軸) with 1 overwritten + distractors
    """
    assert level in (1, 2, 3, 4), f"Invalid level: {level}"
    topic1, topic2 = rng.sample(TOPICS, 2)

    if level in (1, 2):
        n_dist = num_distractors if num_distractors is not None else (
            rng.randint(1, 2) if level == 1 else rng.randint(6, 12)
        )
        mem_slot = rng.choice(["制約", "優先軸"])
        shared_val = rng.choice(SLOT_VALUES[mem_slot])

        slots1 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        slots1[mem_slot] = shared_val
        turn1 = _build_turn1(rng, topic1, slots1, asked_slot=mem_slot)
        distractors = _make_distractors(rng, n_dist)

        slots2 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        slots2[mem_slot] = shared_val
        remaining = [k for k in SLOT_NAMES if k != mem_slot]
        known2 = rng.choice(remaining)
        missing2 = [k for k in remaining if k != known2][0]
        plan2 = PRIORITY_TO_PLAN[slots2["優先軸"]]

        t2_prompt = rng.choice(CONSULT_TEMPLATES).format(
            topic=topic2, known_text=_format_known(slots2, [known2])
        ) + "\n"
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
            level=level,
            prompt=prompt,
            topic=topic2,
            full_slots=slots2,
            known_slots=[known2],
            memory_slots=[mem_slot],
            updated_slots={},
            missing_slots=[missing2],
            expected_plan=plan2,
            ideal_completion=ideal,
            num_distractors=n_dist,
        )

    if level == 3:
        n_dist = num_distractors if num_distractors is not None else rng.randint(4, 8)
        d_before = max(1, n_dist // 2)
        d_after = max(1, n_dist - d_before)

        mem_slot = rng.choice(["制約", "優先軸"])
        old_val, new_val = rng.sample(SLOT_VALUES[mem_slot], 2)

        slots1 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        slots1[mem_slot] = old_val
        turn1 = _build_turn1(rng, topic1, slots1, asked_slot=mem_slot)

        update_line = rng.choice(UPDATE_TEMPLATES).format(slot=mem_slot, new_val=new_val) + "\n"
        middle = _make_distractors(rng, d_before) + update_line + _make_distractors(rng, d_after)

        slots2 = {k: rng.choice(SLOT_VALUES[k]) for k in SLOT_NAMES}
        slots2[mem_slot] = new_val
        remaining = [k for k in SLOT_NAMES if k != mem_slot]
        known2 = rng.choice(remaining)
        missing2 = [k for k in remaining if k != known2][0]
        plan2 = PRIORITY_TO_PLAN[slots2["優先軸"]]

        t2_prompt = rng.choice(CONSULT_TEMPLATES).format(
            topic=topic2, known_text=_format_known(slots2, [known2])
        ) + "\n"
        prompt = turn1 + middle + t2_prompt
        t2_summary = _build_summary(topic2, slots2, plan2)
        ideal = (
            f"<think>テーマ: {topic2}。既知: {known2}={slots2[known2]}。"
            f"更新記憶: {mem_slot}={new_val}。不足: {missing2}</think>"
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
            updated_slots={mem_slot: old_val},
            missing_slots=[missing2],
            expected_plan=plan2,
            ideal_completion=ideal,
            num_distractors=d_before + d_after,
        )

    # Level 4: Composite 2-slot memory recall (制約 + 優先軸), 1 overwritten + long distractors
    n_dist = num_distractors if num_distractors is not None else rng.randint(6, 12)
    d_before = max(1, n_dist // 2)
    d_after = max(1, n_dist - d_before)

    upd_slot = rng.choice(["制約", "優先軸"])
    keep_slot = "優先軸" if upd_slot == "制約" else "制約"
    old_val, new_val = rng.sample(SLOT_VALUES[upd_slot], 2)
    keep_val = rng.choice(SLOT_VALUES[keep_slot])

    slots1 = {
        "目的": rng.choice(SLOT_VALUES["目的"]),
        upd_slot: old_val,
        keep_slot: keep_val,
    }
    turn1 = _build_turn1(rng, topic1, slots1, asked_slot=keep_slot)
    update_line = rng.choice(UPDATE_TEMPLATES).format(slot=upd_slot, new_val=new_val) + "\n"
    middle = _make_distractors(rng, d_before) + update_line + _make_distractors(rng, d_after)

    slots2 = {
        "目的": rng.choice(SLOT_VALUES["目的"]),
        upd_slot: new_val,
        keep_slot: keep_val,
    }
    plan2 = PRIORITY_TO_PLAN[slots2["優先軸"]]
    t2_prompt = rng.choice(CONSULT_NO_KNOWN_TEMPLATES).format(topic=topic2) + "\n"
    prompt = turn1 + middle + t2_prompt
    t2_summary = _build_summary(topic2, slots2, plan2)
    ideal = (
        f"<think>テーマ: {topic2}。記憶: {keep_slot}={keep_val}。"
        f"更新記憶: {upd_slot}={new_val}。不足: 目的</think>"
        f"<ask>目的</ask>"
        f"<user>目的: {slots2['目的']}</user>"
        f"<think>整理完了。全前提が揃った。</think>"
        f"<summary>{t2_summary}</summary><|eos|>"
    )
    return Episode(
        level=4,
        prompt=prompt,
        topic=topic2,
        full_slots=slots2,
        known_slots=[],
        memory_slots=["制約", "優先軸"],
        updated_slots={upd_slot: old_val},
        missing_slots=["目的"],
        expected_plan=plan2,
        ideal_completion=ideal,
        num_distractors=d_before + d_after,
    )


def tokenize_sft_episode(
    ep: Episode,
    tokenizer: ThinkingTokenizer,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenizes an SFT episode with masked labels (-100 on prompt and <user>...</user>).
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
# 3. Models: Titans (MAC) + Optional Fallback Causal Transformer
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
        seq_len=1024,
        sft_batch_size=4,
        sft_grad_accum=2,
        sft_lr=1.5e-3,
        grpo_lr=1e-4,
    ),
    # Kaggle T4/P100 16GB OOM-safe long-context run (~10.2M params)
    "tiny": ModelPreset(
        dim=256,
        depth=4,
        segment_len=64,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
        seq_len=1024,
        sft_batch_size=2,
        sft_grad_accum=16,
        sft_lr=8e-4,
        grpo_lr=4e-5,
    ),
    # Higher capacity (~22M params, batch_size=1 with grad_accum=32 on 16GB)
    "small": ModelPreset(
        dim=384,
        depth=6,
        segment_len=64,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        seq_len=1024,
        sft_batch_size=1,
        sft_grad_accum=32,
        sft_lr=5e-4,
        grpo_lr=2.5e-5,
    ),
}


class FallbackCausalTransformer(nn.Module):
    """Minimal causal Transformer (optionally with sliding window) for comparison/smoke tests."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        max_seq_len: int = 1280,
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

    window_size = model_cfg["segment_len"] if model_type == "sliding_transformer" else 0
    model = FallbackCausalTransformer(
        num_tokens=model_cfg["num_tokens"],
        dim=model_cfg["dim"],
        depth=model_cfg["depth"],
        max_seq_len=max(1280, model_cfg.get("seq_len", 1024) + 256),
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
# 4. Interactive Rollout & Shaped Reward (Strict Format + Overwrite Awareness)
# ---------------------------------------------------------------------------

ASK_RE = re.compile(r"<ask>([^<>]+)</ask>")
SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class RolloutResult:
    full_ids: list[int]
    action_mask: list[int]
    completion_text: str
    asks_made: list[str]
    reward: float
    correct: bool
    slot_accuracy: float
    ask_exact: bool
    memory_reused: bool
    overwrite_ok: bool
    used_stale_memory: bool
    format_ok: bool


def _parse_summary_strict(summary_body: str) -> tuple[dict[str, str], bool]:
    """
    Parses 'k=v | k=v | ...' and rejects empty segments like '| |' or duplicate keys.
    Returns (parsed_dict, clean_pipe_format).
    """
    parts = summary_body.split("|")
    parsed: dict[str, str] = {}
    clean = len(parts) == 5
    for seg in parts:
        seg = seg.strip()
        if not seg or "=" not in seg:
            clean = False
            continue
        k, v = seg.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v or k in parsed:
            clean = False
        parsed[k] = v
    return parsed, clean


def score_rollout(
    ep: Episode,
    completion_text: str,
    asks_made: list[str],
) -> tuple[float, bool, float, bool, bool, bool, bool, bool]:
    """
    Shaped verifier for Exp 2.5:
      - Strict format (<think>, <summary>, exact 5 pipe-separated fields, no '| |'): +0.20
      - Targeted clarification (asking each missing slot once): up to +0.40
      - Redundant / invalid ask penalties: -0.25 / -0.20
      - Summary slot accuracy across 5 fields: up to +1.00
      - Memory reuse bonus (all memory_slots recalled without re-asking): +0.25
      - Overwrite bonus (updated_slots match latest value): +0.30
      - Stale-memory penalty (summary outputs the OLD overwritten value): -0.35
    """
    has_think = bool(THINK_RE.search(completion_text))
    sum_match = SUMMARY_RE.search(completion_text)
    parsed: dict[str, str] = {}
    clean_pipes = False
    if sum_match is not None:
        parsed, clean_pipes = _parse_summary_strict(sum_match.group(1))
    format_ok = has_think and (sum_match is not None) and clean_pipes

    reward = 0.20 if format_ok else (0.05 if (has_think and sum_match is not None) else 0.0)

    # 1) Clarifying questions
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
        reward += 0.40 * (len(hit_missing) / len(missing_set))
    reward -= 0.25 * redundant_cnt
    reward -= 0.20 * invalid_cnt
    if len(asks_made) > 2:
        reward -= 0.20

    ask_exact = (hit_missing == missing_set) and (redundant_cnt == 0) and (invalid_cnt == 0)

    # 2) Summary slot accuracy
    expected_fields = {
        "テーマ": ep.topic,
        "目的": ep.full_slots["目的"],
        "制約": ep.full_slots["制約"],
        "優先軸": ep.full_slots["優先軸"],
        "方針": ep.expected_plan,
    }
    matched_fields = sum(1 for k, v in expected_fields.items() if parsed.get(k) == v)
    slot_accuracy = matched_fields / len(expected_fields)
    reward += 1.00 * slot_accuracy
    correct = (matched_fields == len(expected_fields)) and format_ok and ask_exact

    # 3) Memory reuse & overwrite checks
    all_mem_matched = all(parsed.get(k) == ep.full_slots[k] for k in ep.memory_slots)
    memory_reused = bool(ep.memory_slots) and all_mem_matched and (not reasked_memory)
    if memory_reused:
        reward += 0.25

    used_stale = False
    overwrite_ok = False
    if ep.updated_slots:
        used_stale = any(parsed.get(k) == old_v for k, old_v in ep.updated_slots.items())
        overwrite_ok = all(parsed.get(k) == ep.full_slots[k] for k in ep.updated_slots) and (not reasked_memory)
        if overwrite_ok:
            reward += 0.30
        if used_stale:
            reward -= 0.35

    return reward, correct, slot_accuracy, ask_exact, memory_reused, overwrite_ok, used_stale, format_ok


@torch.no_grad()
def run_interactive_rollout(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    ep: Episode,
    device: torch.device,
    max_new_tokens: int = 160,
    max_seq_len: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.92,
    greedy: bool = False,
    max_asks: int = 2,
) -> RolloutResult:
    """Autoregressively generates completion while intercepting <ask> and injecting <user>."""
    model.eval()
    prompt_ids = tokenizer.encode(ep.prompt)
    if len(prompt_ids) >= max_seq_len - 16:
        prompt_ids = prompt_ids[-(max_seq_len - 16):]
    seq = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    asks_made: list[str] = []

    steps = 0
    while steps < max_new_tokens and len(seq) < max_seq_len:
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]

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
    reward, correct, slot_acc, ask_exact, mem_reused, overwrite_ok, used_stale, format_ok = score_rollout(
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
        overwrite_ok=overwrite_ok,
        used_stale_memory=used_stale,
        format_ok=format_ok,
    )


# ---------------------------------------------------------------------------
# 5. Stage 1: Curriculum SFT (Deliberately Leaves Headroom for GRPO on L2/L4)
# ---------------------------------------------------------------------------

def sample_sft_episode(progress: float, rng: random.Random) -> Episode:
    """
    SFT focuses on shorter distances (d <= 8) and light L4 exposure so GRPO has
    genuine out-of-distribution / harder-distance headroom to improve upon.
    """
    r = rng.random()
    if progress < 0.35:
        if r < 0.55:
            return generate_episode(1, rng, num_distractors=rng.randint(1, 2))
        if r < 0.85:
            return generate_episode(2, rng, num_distractors=rng.randint(4, 7))
        return generate_episode(3, rng, num_distractors=rng.randint(2, 5))
    if progress < 0.70:
        if r < 0.20:
            return generate_episode(1, rng, num_distractors=rng.randint(1, 2))
        if r < 0.50:
            return generate_episode(2, rng, num_distractors=rng.randint(5, 8))
        if r < 0.85:
            return generate_episode(3, rng, num_distractors=rng.randint(4, 7))
        return generate_episode(4, rng, num_distractors=rng.randint(4, 6))
    # Final SFT phase: still keep distractors <= 8 so long-range (10-16) stays challenging
    if r < 0.15:
        return generate_episode(1, rng, num_distractors=rng.randint(1, 2))
    if r < 0.45:
        return generate_episode(2, rng, num_distractors=rng.randint(6, 8))
    if r < 0.80:
        return generate_episode(3, rng, num_distractors=rng.randint(4, 8))
    return generate_episode(4, rng, num_distractors=rng.randint(4, 8))


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
    warmup = max(1, min(50, steps // 10))

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
                    sample_sft_episode(progress, rng),
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
# 6. Stage 2: GRPO on Harder Distances (d=10..16) & Composite Stress (L4)
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


def sample_grpo_episode(rng: random.Random) -> Episode:
    """Harder RL distribution: long-range L2 (d=10..16), L3 overwrite (d=8..14), L4 composite (d=8..14)."""
    r = rng.random()
    if r < 0.30:
        return generate_episode(2, rng, num_distractors=rng.randint(10, 16))
    if r < 0.60:
        return generate_episode(3, rng, num_distractors=rng.randint(8, 14))
    return generate_episode(4, rng, num_distractors=rng.randint(8, 14))


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
    updated_steps = 0

    for step in range(1, steps + 1):
        ep = sample_grpo_episode(rng)

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

        if std_r < 1e-6:
            if step % log_every == 0 or step == 1 or step == steps:
                print(
                    f"  [GRPO] step {step:4d}/{steps} | L{ep.level}(d={ep.num_distractors:2d}) "
                    f"| r {mean_r:.3f} (ema {running_reward:.3f}) | acc {acc_g:.2f} "
                    f"| upd {updated_steps}/{step} | skip(zero-var)"
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
                f"  [GRPO] step {step:4d}/{steps} | L{ep.level}(d={ep.num_distractors:2d}) "
                f"| r {mean_r:.3f} (ema {running_reward:.3f}) | acc {acc_g:.2f} "
                f"| upd {updated_steps}/{step} | kl {total_kl:.4f} | {elapsed:.1f}s"
            )

    ckpt_path = output_dir / "stage2_grpo_latest.pt"
    torch.save({"stage": "grpo", "steps": steps, "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"[Stage 2: GRPO] Saved checkpoint -> {ckpt_path} (updated {updated_steps}/{steps} steps)")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "updated_steps": updated_steps,
        "total_steps": steps,
        "update_rate": round(updated_steps / max(1, steps), 3),
    }


# ---------------------------------------------------------------------------
# 7. Evaluation: 4-Level Benchmark + Distance Retention Curve (d = 2, 6, 12, 18)
# ---------------------------------------------------------------------------

def evaluate_levels(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    device: torch.device,
    eval_per_level: int = 25,
    max_new_tokens: int = 160,
    seq_len: int = 1024,
    seed: int = 9999,
    show_samples: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    eval_specs = {
        1: ("level_1_short_mem", 2),
        2: ("level_2_long_mem_d12", 12),
        3: ("level_3_overwrite_d8", 8),
        4: ("level_4_composite_d10", 10),
    }
    summary: dict[str, Any] = {}

    for lvl, (label, n_dist) in eval_specs.items():
        correct_cnt = 0
        format_cnt = 0
        ask_exact_cnt = 0
        mem_reuse_cnt = 0
        overwrite_cnt = 0
        stale_cnt = 0
        slot_acc_sum = 0.0
        total_reward = 0.0
        avg_prompt_toks = 0.0
        sample_printed = False

        for _ in range(eval_per_level):
            ep = generate_episode(lvl, rng, num_distractors=n_dist)
            avg_prompt_toks += len(tokenizer.encode(ep.prompt))
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
            overwrite_cnt += int(ro.overwrite_ok)
            stale_cnt += int(ro.used_stale_memory)
            slot_acc_sum += ro.slot_accuracy
            total_reward += ro.reward

            if show_samples and not sample_printed:
                sample_printed = True
                print(
                    f"  [Sample L{lvl} d={n_dist}] mem={ep.memory_slots} upd={ep.updated_slots} "
                    f"| asks={ro.asks_made} | correct={ro.correct} | stale={ro.used_stale_memory}"
                )
                print(f"    output: {ro.completion_text}")

        n = max(1, eval_per_level)
        entry: dict[str, Any] = {
            "avg_prompt_tokens": round(avg_prompt_toks / n, 1),
            "accuracy": round(correct_cnt / n, 3),
            "slot_accuracy": round(slot_acc_sum / n, 3),
            "ask_exact_rate": round(ask_exact_cnt / n, 3),
            "memory_reuse_rate": round(mem_reuse_cnt / n, 3),
            "format_rate": round(format_cnt / n, 3),
            "avg_reward": round(total_reward / n, 3),
        }
        if lvl in (3, 4):
            entry["overwrite_rate"] = round(overwrite_cnt / n, 3)
            entry["stale_memory_rate"] = round(stale_cnt / n, 3)
        summary[label] = entry

    return summary


def evaluate_distance_curve(
    model: nn.Module,
    tokenizer: ThinkingTokenizer,
    device: torch.device,
    distances: tuple[int, ...] = (2, 6, 12, 18),
    eval_per_dist: int = 15,
    max_new_tokens: int = 160,
    seq_len: int = 1024,
    seed: int = 12345,
) -> dict[str, Any]:
    """Measures Level 2 (single-slot long-range recall) accuracy across distractor distances."""
    rng = random.Random(seed)
    curve: dict[str, Any] = {}
    for d in distances:
        correct_cnt = 0
        mem_cnt = 0
        tok_sum = 0.0
        for _ in range(eval_per_dist):
            ep = generate_episode(2, rng, num_distractors=d)
            tok_sum += len(tokenizer.encode(ep.prompt))
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
            mem_cnt += int(ro.memory_reused)
        n = max(1, eval_per_dist)
        curve[f"d={d}"] = {
            "avg_prompt_tokens": round(tok_sum / n, 1),
            "accuracy": round(correct_cnt / n, 3),
            "memory_reuse_rate": round(mem_cnt / n, 3),
        }
    return curve


# ---------------------------------------------------------------------------
# 8. CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 2.5: Long-Range Memory & Overwrite Stress Test for Small Titans (MAC)."
    )
    p.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    p.add_argument(
        "--model_type",
        type=str,
        default="titans",
        choices=["titans", "tiny_transformer", "sliding_transformer"],
    )
    p.add_argument("--stage", type=str, default="all", choices=["sft", "grpo", "all", "eval"])
    p.add_argument("--sft_steps", type=int, default=350)
    p.add_argument("--grpo_steps", type=int, default=80)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--grpo_temp", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--kl_beta", type=float, default=0.04)
    p.add_argument("--eval_per_level", type=int, default=25)
    p.add_argument("--eval_per_dist", type=int, default=15)
    p.add_argument("--sft_batch_size", type=int, default=0)
    p.add_argument("--sft_grad_accum", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="checkpoints/exp2_5_memory_stress")
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

    metrics: dict[str, Any] = {
        "preset": args.preset,
        "model_type": args.model_type,
        "preset_cfg": asdict(preset),
    }

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
        metrics["after_sft"] = evaluate_levels(
            model,
            tokenizer,
            device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_sft"], ensure_ascii=False, indent=2))
        print("\n[Distance Curve after Stage 1: SFT]")
        metrics["distance_curve_sft"] = evaluate_distance_curve(
            model,
            tokenizer,
            device,
            eval_per_dist=args.eval_per_dist,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["distance_curve_sft"], ensure_ascii=False, indent=2))

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
        metrics["after_grpo"] = evaluate_levels(
            model,
            tokenizer,
            device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["after_grpo"], ensure_ascii=False, indent=2))
        print("\n[Distance Curve after Stage 2: GRPO]")
        metrics["distance_curve_grpo"] = evaluate_distance_curve(
            model,
            tokenizer,
            device,
            eval_per_dist=args.eval_per_dist,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["distance_curve_grpo"], ensure_ascii=False, indent=2))

    if args.stage == "eval":
        print("\n[Eval]")
        metrics["eval"] = evaluate_levels(
            model,
            tokenizer,
            device,
            eval_per_level=args.eval_per_level,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["eval"], ensure_ascii=False, indent=2))
        metrics["distance_curve"] = evaluate_distance_curve(
            model,
            tokenizer,
            device,
            eval_per_dist=args.eval_per_dist,
            max_new_tokens=args.max_new_tokens,
            seq_len=preset.seq_len,
        )
        print(json.dumps(metrics["distance_curve"], ensure_ascii=False, indent=2))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
