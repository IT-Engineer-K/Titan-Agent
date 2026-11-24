"""
Experiment 1: Japanese-specialized Titans (MAC) Pretraining from Complete Scratch.

- Zero pretrained models, zero external tokenizers.
- Supports from-scratch Japanese ByteLevel BPE tokenizer (default, vocab=4096)
  and pure UTF-8 byte-level tokenizer (vocab=257).
- Supports Google Colab (L4/A100/T4) and Kaggle (T4/P100, 12h background run).
- Uses `titans-pytorch` (MemoryAsContextTransformer).
- Streams Japanese text from Hugging Face (`hotchpotch/fineweb-2-edu-japanese`)
  or runs completely offline with `--dummy_text` for instant sanity checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from titans_pytorch import MemoryAsContextTransformer


EOS_ID_BYTE = 256
VOCAB_SIZE_BYTE = 257  # 0..255 = raw UTF-8 bytes, 256 = EOS


class ByteTokenizer:
    """Zero-dependency UTF-8 byte-level tokenizer (vocab size = 257)."""

    tokenizer_type: str = "byte"
    vocab_size: int = VOCAB_SIZE_BYTE
    eos_token_id: int = EOS_ID_BYTE

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: list[int]) -> str:
        raw = bytes(i for i in ids if 0 <= i < 256)
        return raw.decode("utf-8", errors="ignore")


class ScratchBPETokenizer:
    """From-scratch ByteLevel BPE tokenizer trained directly on the target corpus."""

    tokenizer_type: str = "bpe"

    def __init__(self, tok) -> None:
        self._tok = tok
        self.eos_token_id: int = int(tok.token_to_id("<|eos|>"))
        self.vocab_size: int = int(tok.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        filtered = [i for i in ids if 0 <= i < self.vocab_size and i != self.eos_token_id]
        return self._tok.decode(filtered, skip_special_tokens=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "ScratchBPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterator[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> "ScratchBPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<|eos|>", "<|pad|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)


DUMMY_CORPUS_JA = [
    "人工知能の長期記憶モジュールは、予測しにくい新しい情報ほど強く記憶し、重要度の低い情報を忘却する仕組みを持っています。",
    "ユーザー: 昨日の作業の続きをお願い。\n思考: 前回のログを確認すると、実験1の事前学習スクリプトを作成していた。まずは設定を確認しよう。\n回答: はい、実験1の続きから進めます。",
    "新しいチャットを作らずに1本の会話で継続する場合、会話履歴だけでなく思考プロセスや実行結果も連続した文脈として保持することが重要になります。",
    "ユーザー: 回答はなるべく簡潔な箇条書きにしてほしい。\n思考: ユーザーは簡潔な箇条書きを好む。今後の応答でもこの指示パターンを維持する。\n回答: 了解しました。今後は簡潔な箇条書きでお答えします。",
    "テスト時学習（Test-Time Memorization）では、推論しながらニューラル記憶の重みを更新し続けることで、極端に長い文脈にも対応できます。",
]


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
    # Fast sanity check (~11M params with vocab=4096)
    "tiny": ModelPreset(
        dim=256,
        depth=4,
        segment_len=64,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
        seq_len=512,
        batch_size=8,
        grad_accum=4,
        lr=8e-4,
    ),
    # Colab T4 / L4 or Kaggle P100 (~25M params with vocab=4096)
    "small": ModelPreset(
        dim=384,
        depth=6,
        segment_len=128,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=32,
        seq_len=1024,
        batch_size=4,
        grad_accum=8,
        lr=5e-4,
    ),
    # Colab A100 or Kaggle 12h background run (~50M params with vocab=4096)
    "medium": ModelPreset(
        dim=512,
        depth=8,
        segment_len=128,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=32,
        seq_len=1024,
        batch_size=4,
        grad_accum=8,
        lr=3e-4,
    ),
}


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


def iter_tokenizer_texts(
    dataset_name: str,
    dataset_config: str,
    use_dummy_text: bool,
    max_docs: int,
    seed: int = 42,
) -> Iterator[str]:
    if use_dummy_text:
        for _ in range(20):
            for text in DUMMY_CORPUS_JA:
                yield text
        return

    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        name=dataset_config if dataset_config else None,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    count = 0
    for item in ds:
        text = item.get("text", "")
        if not text:
            continue
        yield text
        count += 1
        if count >= max_docs:
            break


def build_or_load_tokenizer(
    tokenizer_type: str,
    vocab_size: int,
    output_dir: Path,
    resume_path: Path | None,
    dataset_name: str,
    dataset_config: str,
    use_dummy_text: bool,
    tokenizer_train_docs: int,
    seed: int,
) -> ByteTokenizer | ScratchBPETokenizer:
    if tokenizer_type == "byte":
        return ByteTokenizer()

    # Reuse existing tokenizer.json from output_dir or resume checkpoint directory
    candidates = [output_dir / "tokenizer.json"]
    if resume_path is not None:
        candidates.append(resume_path.parent / "tokenizer.json")
    for cand in candidates:
        if cand.exists():
            tok = ScratchBPETokenizer.load(cand)
            if cand != output_dir / "tokenizer.json":
                tok.save(output_dir / "tokenizer.json")
            print(f"[Tokenizer] Loaded existing BPE tokenizer from {cand} (vocab_size={tok.vocab_size})")
            return tok

    t0 = time.time()
    print(
        f"[Tokenizer] Training from-scratch Japanese ByteLevel BPE "
        f"(target vocab={vocab_size}, docs={tokenizer_train_docs if not use_dummy_text else 'dummy'})..."
    )
    texts = iter_tokenizer_texts(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        use_dummy_text=use_dummy_text,
        max_docs=tokenizer_train_docs,
        seed=seed,
    )
    tok = ScratchBPETokenizer.train_from_texts(
        texts,
        vocab_size=vocab_size,
        min_frequency=1 if use_dummy_text else 2,
    )
    save_path = output_dir / "tokenizer.json"
    tok.save(save_path)
    print(f"[Tokenizer] Saved to {save_path} (vocab_size={tok.vocab_size}, {time.time() - t0:.1f}s)")
    return tok


def token_stream(
    dataset_name: str,
    dataset_config: str,
    tokenizer: ByteTokenizer | ScratchBPETokenizer,
    seq_len: int,
    use_dummy_text: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> Iterator[list[int]]:
    """Streams Japanese documents and packs them into fixed-length token chunks."""
    chunk_len = seq_len + 1
    buffer: list[int] = []

    if use_dummy_text:
        idx = 0
        while True:
            text = DUMMY_CORPUS_JA[idx % len(DUMMY_CORPUS_JA)]
            idx += 1
            buffer.extend(tokenizer.encode(text))
            buffer.append(tokenizer.eos_token_id)
            while len(buffer) >= chunk_len:
                yield buffer[:chunk_len]
                buffer = buffer[chunk_len:]

    from datasets import load_dataset

    epoch = 0
    while True:
        ds = load_dataset(
            dataset_name,
            name=dataset_config if dataset_config else None,
            split="train",
            streaming=True,
        )
        if shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer)

        for item in ds:
            text = item.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text)
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(tokenizer.eos_token_id)

            while len(buffer) >= chunk_len:
                yield buffer[:chunk_len]
                buffer = buffer[chunk_len:]

        epoch += 1


def batch_iterator(stream: Iterator[list[int]], batch_size: int) -> Iterator[torch.Tensor]:
    while True:
        batch = [next(stream) for _ in range(batch_size)]
        yield torch.tensor(batch, dtype=torch.long)


def cosine_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@torch.no_grad()
def generate_sample(
    model: MemoryAsContextTransformer,
    tokenizer: ByteTokenizer | ScratchBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt) or [tokenizer.eos_token_id]
    prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    # Note: titans_pytorch's model.sample(prompt, seq_len) takes TOTAL sequence length.
    total_seq_len = len(ids) + max_new_tokens
    sampled = model.sample(prompt_tensor, total_seq_len)
    out_ids = sampled[0].tolist()
    model.train()
    return tokenizer.decode(out_ids)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: MemoryAsContextTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_cfg: dict,
    tokenizer_type: str,
    loss_val: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_step_{step:06d}.pt"
    latest_path = output_dir / "latest.pt"
    payload = {
        "step": step,
        "loss": loss_val,
        "model_cfg": model_cfg,
        "tokenizer_type": tokenizer_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    return ckpt_path


def resolve_resume_path(resume: str, output_dir: Path) -> Path | None:
    if not resume:
        return None
    if resume == "auto":
        cand = output_dir / "latest.pt"
        return cand if cand.exists() else None
    cand = Path(resume)
    return cand if cand.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1: Pure from-scratch Japanese Titans (MAC) pretraining.")
    parser.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "byte"])
    parser.add_argument("--vocab_size", type=int, default=4096, help="Target BPE vocab size (ignored for byte)")
    parser.add_argument("--tokenizer_train_docs", type=int, default=10000, help="Docs to stream for BPE training")
    parser.add_argument("--dataset", type=str, default="hotchpotch/fineweb-2-edu-japanese")
    parser.add_argument("--dataset_config", type=str, default="sample_10BT")
    parser.add_argument("--dummy_text", action="store_true", help="Use built-in Japanese sample text (100% offline)")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--sample_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="checkpoints/exp1_bpe")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint .pt path or 'auto' to resume from latest.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    preset = PRESETS[args.preset]
    seq_len = args.seq_len or preset.seq_len
    batch_size = args.batch_size or preset.batch_size
    grad_accum = args.grad_accum or preset.grad_accum
    max_lr = args.lr or preset.lr

    device, amp_dtype = get_device_and_dtype()
    use_amp = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume_path(args.resume, output_dir)

    print(
        f"[Env] device={device}, amp_dtype={amp_dtype}, preset={args.preset}, "
        f"tokenizer={args.tokenizer}, dummy_text={args.dummy_text}"
    )
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    tokenizer = build_or_load_tokenizer(
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        output_dir=output_dir,
        resume_path=resume_path,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        use_dummy_text=args.dummy_text,
        tokenizer_train_docs=args.tokenizer_train_docs,
        seed=args.seed,
    )

    model_cfg = {
        "num_tokens": tokenizer.vocab_size,
        "dim": preset.dim,
        "depth": preset.depth,
        "segment_len": preset.segment_len,
        "num_persist_mem_tokens": preset.num_persist_mem_tokens,
        "num_longterm_mem_tokens": preset.num_longterm_mem_tokens,
    }

    ckpt = None
    start_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        if "model_cfg" in ckpt:
            model_cfg = ckpt["model_cfg"]
        start_step = int(ckpt["step"])
        print(f"[Resume] Loaded {resume_path} at step {start_step}")

    model = MemoryAsContextTransformer(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] MemoryAsContextTransformer params: {total_params / 1e6:.2f}M (vocab={model_cfg['num_tokens']})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    run_meta = {
        "preset": args.preset,
        "preset_values": asdict(preset),
        "model_cfg": model_cfg,
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "dummy_text": args.dummy_text,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_lr": max_lr,
        "max_steps": args.max_steps,
    }
    (output_dir / "config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Shift stream seed slightly when resuming so we don't repeat the exact same initial documents
    stream_seed = args.seed + start_step
    stream = token_stream(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        seq_len=seq_len,
        use_dummy_text=args.dummy_text,
        seed=stream_seed,
    )
    batches = batch_iterator(stream, batch_size=batch_size)

    model.train()
    t0 = time.time()
    tokens_per_step = batch_size * grad_accum * seq_len
    print(f"[Train] Starting from step {start_step} to {args.max_steps} ({tokens_per_step:,} tokens/step)")

    sample_prompts = [
        "人工知能の長期記憶において重要なのは、",
        "ユーザー: 昨日の作業の続きをお願い。\n思考: ",
    ]

    last_loss = float("nan")
    for step in range(start_step + 1, args.max_steps + 1):
        lr = cosine_lr(step - 1, args.warmup_steps, args.max_steps, max_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            x = next(batches).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(x, return_loss=True)
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

        last_loss = accum_loss

        if step % args.log_every == 0 or step == start_step + 1:
            elapsed = time.time() - t0
            steps_done = step - start_step
            tok_per_sec = (steps_done * tokens_per_step) / max(elapsed, 1e-6)
            ppl = math.exp(min(accum_loss, 20.0))
            metric = f"bpb {accum_loss / math.log(2):.3f}" if tokenizer.tokenizer_type == "byte" else f"ppl {ppl:.2f}"
            print(
                f"step {step:6d}/{args.max_steps} | loss {accum_loss:.4f} | {metric} | "
                f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
            )

        if step % args.sample_every == 0:
            print("-" * 60)
            for p in sample_prompts:
                gen = generate_sample(model, tokenizer, p, device=device, max_new_tokens=80)
                print(f"[Prompt] {p}\n[Output] {gen}\n")
            print("-" * 60)

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_cfg=model_cfg,
                tokenizer_type=tokenizer.tokenizer_type,
                loss_val=last_loss,
            )
            print(f"[Checkpoint] Saved to {ckpt_path}")

    print("[Done] Experiment 1 training finished.")


if __name__ == "__main__":
    main()
