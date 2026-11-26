"""
Phase 2 Training Script: 2-Stage SFT for Titans (MAC) Adapter on Frozen Japanese LLM.

Applies the findings from Exp 1 - Exp 3.5 directly to a real Japanese LLM
(default: Qwen/Qwen2.5-0.5B-Instruct, or --mock for fast local CPU smoke tests):

  1. Memory-Required Training Design:
       Early turns (user profile, goals, constraints, mid-stream updates, and
       distractors) are streamed ONLY into the Titans Neural Memory fast weights M_t,
       and intentionally omitted from the final short-term prompt.
       Thus the frozen LLM must learn to read the prepended MAC memory tokens
       to recall and overwrite user premises accurately.
  2. 2-Stage Curriculum (Exp 3.5 finding):
       - Stage 1 (30%): Short single-segment memory + <think> rehearsal warmup.
       - Stage 2 (70%): Multi-segment long-stream + premise overwrites + distractors.
  3. Turn-level CoT Rehearsal (Exp 3.5 winner):
       Targets include concise <think>...</think> premise rehearsal before the
       natural Japanese reply.

Usage:
  # Local CPU smoke test (~5 seconds, zero downloads):
  python phase2_llm_titans_chat/train_mac_adapter.py --mock --steps 12 --eval_episodes 4

  # Kaggle T4 / Colab GPU run (~20-30 min):
  python phase2_llm_titans_chat/train_mac_adapter.py --model_name Qwen/Qwen2.5-0.5B-Instruct --steps 300
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from titans_mac_adapter import (
    TitansMACLLM,
    TitansMemoryConfig,
    load_base_llm_and_tokenizer,
)


# ---------------------------------------------------------------------------
# 1. Natural Japanese 1-Stream Memory & Overwrite Episode Generator
# ---------------------------------------------------------------------------

NAMES = ["ケンタ", "ユウキ", "ミサキ", "ハルカ", "タクミ", "アオイ", "リョウ", "サクラ"]
THEMES = ["英語学習", "アプリ開発", "副業ブログ", "資格勉強", "運動習慣", "AI研究", "デザイン学習", "自動化ツール作り"]
GOALS = ["毎日続けること", "3か月以内に形にすること", "基礎を固めること", "小さく公開すること", "収入につなげること"]
CONSTRAINTS = ["平日は30分だけ", "週末にまとめて作業", "予算は無料枠のみ", "夜しか時間がない", "スマホ中心で進めたい"]
STYLES = ["簡潔に要点だけ", "丁寧にステップ順で", "具体例を交えて", "背中を押す感じで"]

DISTRACTOR_TURNS = [
    "ユーザー: 今日は少し天気がいいですね。ちょっと一息ついてます。\nアシスタント: いいですね、無理のないペースでリフレッシュしてください。",
    "ユーザー: さっきコーヒーを淹れました。作業環境を整えています。\nアシスタント: 環境が整うと集中しやすいですよね。準備ができたら進みましょう。",
    "ユーザー: そういえば昨日読んだ記事が面白かったです。\nアシスタント: 良い刺激になりますね。気になった視点があればメモしておくと役立ちます。",
    "ユーザー: ちょっと別の用事を片付けてきます。\nアシスタント: 了解です！戻ってきたらまた続きを話しましょう。",
    "ユーザー: メモの整理をしていました。少しずつ見通しが立ってきました。\nアシスタント: 整理が進んで何よりです。今の流れを大事にしましょう。",
]


@dataclass
class JapaneseMemoryEpisode:
    memory_segments: list[str]
    final_prompt: str
    target_response: str
    latest_slots: dict[str, str]
    stale_slots: dict[str, str]
    has_overwrite: bool


def generate_japanese_episode(
    rng: random.Random,
    stage: int = 2,
    force_overwrite: bool | None = None,
) -> JapaneseMemoryEpisode:
    name = rng.choice(NAMES)
    theme = rng.choice(THEMES)
    goal = rng.choice(GOALS)
    constraint = rng.choice(CONSTRAINTS)
    style = rng.choice(STYLES)

    latest = {
        "名前": name,
        "テーマ": theme,
        "目標": goal,
        "制約": constraint,
        "好み": style,
    }
    stale: dict[str, str] = {}

    seg1 = (
        f"ユーザー: はじめまして、{name}です。今取り組んでいるのは{theme}で、"
        f"目標は「{goal}」、制約は「{constraint}」です。アドバイスは{style}お願いします。\n"
        f"アシスタント: <think>記憶: 名前={name} / テーマ={theme} / 目標={goal} / 制約={constraint} / 好み={style}</think>\n"
        f"{name}さん、よろしくお願いします！{theme}を「{constraint}」の条件で進められるように一緒に考えていきましょう。"
    )
    segments = [seg1]

    if stage == 1:
        do_overwrite = False if force_overwrite is None else force_overwrite
        n_dist = rng.randint(0, 1)
    else:
        do_overwrite = (rng.random() < 0.55) if force_overwrite is None else force_overwrite
        n_dist = rng.randint(1, 3)

    for _ in range(n_dist):
        segments.append(rng.choice(DISTRACTOR_TURNS))

    if do_overwrite:
        slot_to_update = rng.choice(["制約", "目標", "テーマ"])
        old_val = latest[slot_to_update]
        pool = CONSTRAINTS if slot_to_update == "制約" else (GOALS if slot_to_update == "目標" else THEMES)
        new_val = rng.choice([x for x in pool if x != old_val])
        stale[slot_to_update] = old_val
        latest[slot_to_update] = new_val
        upd_seg = (
            f"ユーザー: ちょっと状況が変わりました。{slot_to_update}を「{new_val}」に変更したいです。\n"
            f"アシスタント: <think>更新記憶: {slot_to_update}={new_val}（旧:{old_val}は破棄）</think>\n"
            f"了解しました！{slot_to_update}を「{new_val}」に更新して覚えておきますね。"
        )
        segments.append(upd_seg)
        if stage == 2:
            segments.append(rng.choice(DISTRACTOR_TURNS))

    final_prompt = (
        "ユーザー: 前に伝えた私の名前と、今のテーマ・目標・制約を踏まえて、次の一歩を提案してくれますか？\n"
        "アシスタント: "
    )
    target_response = (
        f"<think>記憶リハーサル: 名前={latest['名前']} | テーマ={latest['テーマ']} | "
        f"目標={latest['目標']} | 制約={latest['制約']} | 好み={latest['好み']}</think>\n"
        f"{latest['名前']}さん、現在の前提（テーマ: {latest['テーマ']} / 目標: {latest['目標']} / "
        f"制約: {latest['制約']}）に合わせて{latest['好み']}提案します。"
        f"まずは「{latest['制約']}」の中で{latest['テーマ']}を小さく1回試し、「{latest['目標']}」へつなげましょう。"
    )

    return JapaneseMemoryEpisode(
        memory_segments=segments,
        final_prompt=final_prompt,
        target_response=target_response,
        latest_slots=latest,
        stale_slots=stale,
        has_overwrite=do_overwrite,
    )


# ---------------------------------------------------------------------------
# 2. Training & Evaluation
# ---------------------------------------------------------------------------

def score_japanese_reply(ep: JapaneseMemoryEpisode, reply: str) -> dict[str, Any]:
    recalled = 0
    for key in ["名前", "テーマ", "目標", "制約"]:
        if ep.latest_slots[key] in reply:
            recalled += 1
    recall_rate = recalled / 4.0

    overwrite_ok = True
    used_stale = False
    if ep.has_overwrite:
        for k, old_v in ep.stale_slots.items():
            new_v = ep.latest_slots[k]
            if new_v not in reply:
                overwrite_ok = False
            if old_v in reply and new_v not in reply:
                used_stale = True

    format_ok = "<think>" in reply and "</think>" in reply
    return {
        "recall_rate": recall_rate,
        "overwrite_ok": overwrite_ok,
        "used_stale": used_stale,
        "format_ok": format_ok,
    }


def train_adapter(
    wrapper: TitansMACLLM,
    tokenizer: Any,
    device: torch.device,
    steps: int,
    grad_accum: int,
    lr: float,
    output_dir: Path,
    log_every: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(wrapper.adapter.parameters(), lr=lr, weight_decay=0.01)
    stage1_steps = max(1, int(steps * 0.30))

    wrapper.adapter.train()
    t0 = time.time()
    last_loss = 0.0

    for step in range(1, steps + 1):
        stage = 1 if step <= stage1_steps else 2
        if step <= max(1, int(steps * 0.05)):
            cur_lr = lr * (step / max(1, int(steps * 0.05)))
        else:
            prog = (step - max(1, int(steps * 0.05))) / max(1, steps - max(1, int(steps * 0.05)))
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        avg_surprise = 0.0

        for _ in range(grad_accum):
            ep = generate_japanese_episode(rng, stage=stage)
            mem_tensors = [
                torch.tensor([tokenizer.encode(seg, add_special_tokens=False)], dtype=torch.long, device=device)
                for seg in ep.memory_segments
            ]
            prompt_ids = tokenizer.encode(ep.final_prompt, add_special_tokens=False)
            resp_ids = tokenizer.encode(ep.target_response, add_special_tokens=False) + [tokenizer.eos_token_id]
            full_ids = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + resp_ids

            target_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            labels_t = torch.tensor([labels], dtype=torch.long, device=device)

            loss, _, stats = wrapper.forward_with_memory(mem_tensors, target_t, labels_t)
            (loss / grad_accum).backward()
            accum_loss += float(loss.item()) / grad_accum
            avg_surprise += stats["surprise"] / grad_accum

        torch.nn.utils.clip_grad_norm_(wrapper.adapter.parameters(), 1.0)
        optimizer.step()
        last_loss = accum_loss

        if step == 1 or step % log_every == 0 or step == steps:
            tag = "S1_warm" if stage == 1 else "S2_over"
            print(
                f"  [MAC | {tag}] step {step:4d}/{steps} | loss {accum_loss:.4f} "
                f"| surprise {avg_surprise:.4f} | lr {cur_lr:.2e} | {time.time() - t0:.1f}s"
            )

    ckpt_path = output_dir / "titans_mac_adapter.pt"
    wrapper.save_adapter(ckpt_path, extra={"steps": steps, "final_loss": round(last_loss, 4)})
    return {
        "steps": steps,
        "final_loss": round(last_loss, 4),
        "train_seconds": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path),
    }


@torch.no_grad()
def evaluate_adapter(
    wrapper: TitansMACLLM,
    tokenizer: Any,
    device: torch.device,
    eval_episodes: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    wrapper.adapter.eval()
    rng = random.Random(seed)
    recall_sum = 0.0
    overwrite_hits = 0
    overwrite_total = 0
    stale_hits = 0
    fmt_hits = 0
    sample_info: dict[str, Any] = {}

    for idx in range(eval_episodes):
        force_ov = (idx % 2 == 1)
        ep = generate_japanese_episode(rng, stage=2, force_overwrite=force_ov)
        state = wrapper.init_memory_state(batch_size=1, device=device)
        for seg in ep.memory_segments:
            seg_t = torch.tensor([tokenizer.encode(seg, add_special_tokens=False)], dtype=torch.long, device=device)
            state, _ = wrapper.memorize_segment(seg_t, state)

        prompt_ids = tokenizer.encode(ep.final_prompt, add_special_tokens=False)
        gen_ids = wrapper.generate_with_memory(
            prompt_ids=prompt_ids,
            state=state,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            device=device,
        )
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
        sc = score_japanese_reply(ep, reply)

        recall_sum += sc["recall_rate"]
        fmt_hits += int(sc["format_ok"])
        if ep.has_overwrite:
            overwrite_total += 1
            overwrite_hits += int(sc["overwrite_ok"])
            stale_hits += int(sc["used_stale"])

        if idx == 1 or (idx == 0 and eval_episodes == 1):
            sample_info = {
                "latest_slots": ep.latest_slots,
                "stale_slots": ep.stale_slots,
                "num_memory_segments": len(ep.memory_segments),
                "reply_preview": reply[:240],
                "score": sc,
            }

    return {
        "eval_episodes": eval_episodes,
        "memory_recall_rate": round(recall_sum / max(1, eval_episodes), 4),
        "overwrite_rate": round(overwrite_hits / max(1, overwrite_total), 4),
        "stale_rate": round(stale_hits / max(1, overwrite_total), 4),
        "format_ok_rate": round(fmt_hits / max(1, eval_episodes), 4),
        "sample": sample_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titans MAC Adapter on Frozen Japanese LLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mock", action="store_true", help="Use tiny mock LM for fast CPU smoke tests")
    parser.add_argument("--mem_dim", type=int, default=256)
    parser.add_argument("--num_persist", type=int, default=4)
    parser.add_argument("--num_longterm", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2_mac_adapter")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device={device} | model={'TinyMockCausalLM' if args.mock else args.model_name}")
    base_model, tokenizer, hidden_size = load_base_llm_and_tokenizer(
        args.model_name, use_mock=args.mock, device=device
    )

    mem_cfg = TitansMemoryConfig(
        hidden_size=hidden_size,
        mem_dim=min(args.mem_dim, hidden_size) if args.mock else args.mem_dim,
        num_persist_mem_tokens=args.num_persist,
        num_longterm_mem_tokens=args.num_longterm,
    )
    wrapper = TitansMACLLM(base_model, mem_cfg, freeze_base=True).to(device)
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    print(f"Params: trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M")

    train_stats = train_adapter(
        wrapper=wrapper,
        tokenizer=tokenizer,
        device=device,
        steps=args.steps,
        grad_accum=args.grad_accum,
        lr=args.lr,
        output_dir=output_dir,
        log_every=args.log_every,
        seed=args.seed,
    )
    eval_stats = evaluate_adapter(
        wrapper=wrapper,
        tokenizer=tokenizer,
        device=device,
        eval_episodes=args.eval_episodes,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed + 999,
    )

    metrics = {"model": "mock" if args.mock else args.model_name, "train": train_stats, "eval": eval_stats}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Phase 2 MAC Adapter Eval ===")
    print(json.dumps(eval_stats, ensure_ascii=False, indent=2))
    print(f"[Done] Saved to {output_dir}")


if __name__ == "__main__":
    main()
