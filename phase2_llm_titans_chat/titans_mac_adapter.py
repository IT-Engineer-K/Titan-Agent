"""
Phase 2: Titans (MAC: Memory as a Context) Adapter for Frozen Japanese LLMs.

Attaches a genuine test-time-updating Titans Neural Memory (arXiv:2501.00663)
to any pretrained HuggingFace CausalLM (e.g., Qwen/Qwen2.5-0.5B-Instruct)
without modifying base LLM weights:

  - Persistent Memory: learnable task/persona soft-prompt tokens.
  - Neural Long-Term Memory (Fast Weights W1, W2 + Momentum S1, S2):
      Updated at test time on each conversation segment via analytical MSE
      surprise gradients with learned momentum (eta), learning rate (theta),
      and adaptive forget gate (alpha).
      Fully differentiable during MAC adapter meta-training AND works under
      torch.no_grad() at inference time (fp16/bf16 safe).
  - Memory as a Context (MAC):
      Queries fast weights M_t using current prompt embeddings to synthesize
      soft memory tokens prepended to the frozen LLM's input embeddings:
        [Persistent Memory | Retrieved Titans Memory | Active Window Tokens]
  - Session Persistence:
      Saves/loads Neural Memory fast weights M_t + 1-stream history to disk
      so the agent continues a single lifelong conversation across restarts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitansMemoryConfig:
    hidden_size: int = 896  # Qwen2.5-0.5B default; auto-inferred when loading HF model
    mem_dim: int = 256
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 8
    max_grad_norm: float = 1.0
    base_lr_scale: float = 0.15
    max_active_turns: int = 3


@dataclass
class TitansMemoryState:
    """Mutable test-time neural memory state (fast weights + momentum)."""

    w1: torch.Tensor  # [B, mem_dim, mem_dim]
    w2: torch.Tensor  # [B, mem_dim, mem_dim]
    s1: torch.Tensor  # [B, mem_dim, mem_dim]
    s2: torch.Tensor  # [B, mem_dim, mem_dim]
    num_updates: int = 0
    last_surprise: float = 0.0
    last_forget_rate: float = 0.0

    def detach_(self) -> "TitansMemoryState":
        self.w1 = self.w1.detach()
        self.w2 = self.w2.detach()
        self.s1 = self.s1.detach()
        self.s2 = self.s2.detach()
        return self

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "TitansMemoryState":
        return TitansMemoryState(
            w1=self.w1.to(device=device, dtype=dtype or self.w1.dtype),
            w2=self.w2.to(device=device, dtype=dtype or self.w2.dtype),
            s1=self.s1.to(device=device, dtype=dtype or self.s1.dtype),
            s2=self.s2.to(device=device, dtype=dtype or self.s2.dtype),
            num_updates=self.num_updates,
            last_surprise=self.last_surprise,
            last_forget_rate=self.last_forget_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "w1": self.w1.detach().cpu(),
            "w2": self.w2.detach().cpu(),
            "s1": self.s1.detach().cpu(),
            "s2": self.s2.detach().cpu(),
            "num_updates": self.num_updates,
            "last_surprise": self.last_surprise,
            "last_forget_rate": self.last_forget_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], device: torch.device) -> "TitansMemoryState":
        return cls(
            w1=data["w1"].to(device),
            w2=data["w2"].to(device),
            s1=data["s1"].to(device),
            s2=data["s2"].to(device),
            num_updates=int(data.get("num_updates", 0)),
            last_surprise=float(data.get("last_surprise", 0.0)),
            last_forget_rate=float(data.get("last_forget_rate", 0.0)),
        )


class TitansMACAdapter(nn.Module):
    """
    Titans Neural Long-Term Memory + Memory-as-a-Context (MAC) prefix generator.

    All internal fast-weight operations run in float32 for numerical stability,
    then cast to the base LLM's embedding dtype on output.
    """

    def __init__(self, cfg: TitansMemoryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        d = cfg.mem_dim

        # Input normalization from frozen LLM embedding space
        self.in_norm = nn.LayerNorm(h)

        # Associative key/value/query projections
        self.k_proj = nn.Linear(h, d, bias=False)
        self.v_proj = nn.Linear(h, d, bias=False)
        self.q_proj = nn.Linear(h, d, bias=False)

        # Learned test-time update gates (from segment summary)
        # alpha: forget gate (biased small ~0.05 so memories persist)
        # eta:   momentum retention (~0.5)
        # theta: surprise step size (positive via softplus)
        self.gate_proj = nn.Linear(h, 3)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias.copy_(torch.tensor([-3.0, 0.0, -1.0]))

        # Learned initial fast weights M_0 (meta-learned initialization)
        self.w1_init = nn.Parameter(torch.empty(d, d))
        self.w2_init = nn.Parameter(torch.empty(d, d))
        nn.init.normal_(self.w1_init, mean=0.0, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.w2_init, mean=0.0, std=1.0 / math.sqrt(d))

        # Persistent memory tokens (always prepended to context)
        self.persist_mem = nn.Parameter(
            torch.randn(1, cfg.num_persist_mem_tokens, h) * 0.02
        )

        # Learned retrieval queries (combined with current prompt summary)
        self.read_queries = nn.Parameter(
            torch.randn(1, cfg.num_longterm_mem_tokens, d) * 0.02
        )
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, h, bias=False)
        # Start with small residual scale so frozen LLM isn't disrupted early in training
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def init_state(self, batch_size: int, device: torch.device) -> TitansMemoryState:
        w1 = self.w1_init.float().unsqueeze(0).expand(batch_size, -1, -1).clone()
        w2 = self.w2_init.float().unsqueeze(0).expand(batch_size, -1, -1).clone()
        s1 = torch.zeros_like(w1)
        s2 = torch.zeros_like(w2)
        return TitansMemoryState(w1=w1, w2=w2, s1=s1, s2=s2, num_updates=0)

    def _memory_forward(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """2-layer SiLU MLP fast memory: pred = silu(x @ W1) @ W2."""
        z = torch.bmm(x, w1)
        sig = torch.sigmoid(z)
        h = z * sig
        pred = torch.bmm(h, w2)
        return pred, z, sig, h

    def update_memory(
        self,
        segment_embeds: torch.Tensor,
        state: TitansMemoryState,
        segment_mask: torch.Tensor | None = None,
    ) -> tuple[TitansMemoryState, dict[str, float]]:
        """
        Test-time memorization step on a segment of embeddings [B, S, H].
        Computes analytical surprise gradient dL/dM and updates (W1, W2, S1, S2).
        """
        x_f32 = self.in_norm(segment_embeds.float())
        k = F.normalize(self.k_proj(x_f32), dim=-1)  # [B, S, D]
        v = self.v_proj(x_f32)  # [B, S, D]

        if segment_mask is not None:
            mask_f = segment_mask.float().unsqueeze(-1)  # [B, S, 1]
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            seg_summary = (x_f32 * mask_f).sum(dim=1) / denom.squeeze(1)  # [B, H]
        else:
            mask_f = None
            seg_summary = x_f32.mean(dim=1)  # [B, H]

        gates = self.gate_proj(seg_summary)  # [B, 3]
        alpha = torch.sigmoid(gates[:, 0]).view(-1, 1, 1)  # forget gate
        eta = torch.sigmoid(gates[:, 1]).view(-1, 1, 1)  # momentum decay
        theta = F.softplus(gates[:, 2]).view(-1, 1, 1) * self.cfg.base_lr_scale

        # Forward pass through current fast weights M_t
        pred, z, sig, h = self._memory_forward(k, state.w1, state.w2)
        diff = pred - v
        if mask_f is not None:
            diff = diff * mask_f
            token_count = mask_f.sum(dim=(1, 2)).clamp_min(1.0).view(-1, 1, 1)
        else:
            token_count = float(k.shape[1])

        surprise_per_sample = 0.5 * (diff.pow(2).sum(dim=(1, 2)) / (
            token_count.view(-1) * k.shape[-1] if isinstance(token_count, torch.Tensor) else token_count * k.shape[-1]
        ))

        # Analytical gradient of 0.5 * mean(||pred - v||^2) w.r.t. W1, W2
        err = diff / token_count  # [B, S, D]
        dw2 = torch.bmm(h.transpose(1, 2), err)  # [B, D, D]
        dh = torch.bmm(err, state.w2.transpose(1, 2))  # [B, S, D]
        dsilu = sig * (1.0 + z * (1.0 - sig))
        dz = dh * dsilu
        dw1 = torch.bmm(k.transpose(1, 2), dz)  # [B, D, D]

        # Per-sample gradient norm clipping for fp16/meta-training stability
        max_n = self.cfg.max_grad_norm
        dw1_norm = dw1.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        dw2_norm = dw2.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        dw1 = dw1 * (max_n / dw1_norm).clamp(max=1.0)
        dw2 = dw2 * (max_n / dw2_norm).clamp(max=1.0)

        # Titans momentum + adaptive forget update
        new_s1 = eta * state.s1 - theta * dw1
        new_s2 = eta * state.s2 - theta * dw2
        new_w1 = (1.0 - alpha) * state.w1 + new_s1
        new_w2 = (1.0 - alpha) * state.w2 + new_s2

        surprise_val = float(surprise_per_sample.mean().detach().item())
        forget_val = float(alpha.mean().detach().item())

        new_state = TitansMemoryState(
            w1=new_w1,
            w2=new_w2,
            s1=new_s1,
            s2=new_s2,
            num_updates=state.num_updates + 1,
            last_surprise=surprise_val,
            last_forget_rate=forget_val,
        )
        stats = {
            "surprise": surprise_val,
            "forget_rate": forget_val,
            "lr_theta": float(theta.mean().detach().item()),
        }
        return new_state, stats

    def retrieve_prefix(
        self,
        prompt_embeds: torch.Tensor,
        state: TitansMemoryState,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Queries M_t using current prompt context and returns MAC prefix embeddings:
          [persist_mem (P), retrieved_mem (M)] of shape [B, P + M, H].
        """
        bsz = prompt_embeds.shape[0]
        x_f32 = self.in_norm(prompt_embeds.float())
        prompt_q = self.q_proj(x_f32.mean(dim=1, keepdim=True))  # [B, 1, D]

        # Combine prompt-conditioned query with learned multi-slot read queries
        queries = F.normalize(
            self.read_queries.float().expand(bsz, -1, -1) + prompt_q, dim=-1
        )  # [B, M, D]

        retrieved, _, _, _ = self._memory_forward(queries, state.w1, state.w2)
        retrieved = self.out_proj(self.out_norm(retrieved)) * self.out_scale  # [B, M, H]

        persist = self.persist_mem.float().expand(bsz, -1, -1)  # [B, P, H]
        prefix = torch.cat([persist, retrieved], dim=1)  # [B, P + M, H]
        return prefix.to(dtype=out_dtype)


class TinyMockTokenizer:
    """Zero-download UTF-8 character tokenizer for fast local CPU smoke tests."""

    def __init__(self, chars: str = "") -> None:
        base_tokens = ["<|pad|>", "<|eos|>", "<|unk|>"]
        default_chars = (
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっー、"
            "。！？：「」（）\n 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/<>=-_|"
            "ユーザーアシスタント前提記憶更新変更目標制約名前好口調仕事進捗整理思考確認中毎日週末無料予算時間英語学習開発副業執筆資格運動習慣"
        )
        vocab_list = list(base_tokens)
        seen = set(vocab_list)
        for ch in default_chars + chars:
            if ch not in seen:
                vocab_list.append(ch)
                seen.add(ch)
        self.id_to_tok = vocab_list
        self.tok_to_id = {t: i for i, t in enumerate(vocab_list)}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.tok_to_id.get(ch, self.unk_token_id) for ch in text]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out = []
        for idx in ids:
            if skip_special_tokens and idx in (self.pad_token_id, self.eos_token_id):
                continue
            if 0 <= idx < len(self.id_to_tok):
                out.append(self.id_to_tok[idx])
        return "".join(out)


class TinyMockCausalLM(nn.Module):
    """Lightweight Transformer LM with HF-compatible inputs_embeds interface for CPU tests."""

    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(1024, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape
        pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)
        x = inputs_embeds + self.pos_emb(pos.clamp(max=1023))
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(x, mask=causal)
        logits = self.lm_head(self.norm(h))
        return type("Out", (), {"logits": logits})()


class TitansMACLLM(nn.Module):
    """
    Wraps a frozen base CausalLM with a trainable TitansMACAdapter.
    """

    def __init__(
        self,
        base_model: nn.Module,
        mem_cfg: TitansMemoryConfig,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
        self.mem_cfg = mem_cfg
        self.adapter = TitansMACAdapter(mem_cfg)

    @property
    def prefix_len(self) -> int:
        return self.mem_cfg.num_persist_mem_tokens + self.mem_cfg.num_longterm_mem_tokens

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base_model.get_input_embeddings()
        return emb(input_ids)

    def init_memory_state(self, batch_size: int = 1, device: torch.device | None = None) -> TitansMemoryState:
        dev = device or next(self.adapter.parameters()).device
        return self.adapter.init_state(batch_size=batch_size, device=dev)

    def memorize_segment(
        self,
        segment_ids: torch.Tensor,
        state: TitansMemoryState,
        segment_mask: torch.Tensor | None = None,
    ) -> tuple[TitansMemoryState, dict[str, float]]:
        with torch.no_grad():
            seg_embeds = self.embed_ids(segment_ids)
        return self.adapter.update_memory(seg_embeds, state, segment_mask=segment_mask)

    def forward_with_memory(
        self,
        memory_segments: list[torch.Tensor],
        target_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TitansMemoryState, dict[str, float]]:
        """
        End-to-end differentiable forward pass:
          1. Streams each segment in `memory_segments` into Titans Neural Memory M_t.
          2. Retrieves MAC prefix conditioned on `target_ids`.
          3. Runs frozen base LLM on [MAC_prefix, target_embeds] and computes CE loss on `labels`.
        """
        bsz = target_ids.shape[0]
        device = target_ids.device
        state = self.init_memory_state(batch_size=bsz, device=device)

        last_stats = {"surprise": 0.0, "forget_rate": 0.0, "lr_theta": 0.0}
        for seg_ids in memory_segments:
            if seg_ids.shape[1] == 0:
                continue
            state, last_stats = self.memorize_segment(seg_ids, state)

        with torch.no_grad():
            target_embeds = self.embed_ids(target_ids)

        prefix_embeds = self.adapter.retrieve_prefix(
            target_embeds, state, out_dtype=target_embeds.dtype
        )
        full_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(
                (bsz, self.prefix_len), dtype=attention_mask.dtype, device=device
            )
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = None

        outputs = self.base_model(inputs_embeds=full_embeds, attention_mask=full_mask)
        logits = outputs.logits  # [B, prefix_len + T, V]

        # Align logits with target tokens (ignore prefix positions)
        target_logits = logits[:, self.prefix_len - 1 : -1, :]  # predicts target_ids[0..T-1]
        # Shift so position i predicts labels[i+1] matching standard CausalLM convention
        shift_logits = logits[:, self.prefix_len : -1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        _ = target_logits
        return loss, state, last_stats

    @torch.no_grad()
    def generate_with_memory(
        self,
        prompt_ids: list[int],
        state: TitansMemoryState,
        eos_token_id: int,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_p: float = 0.9,
        device: torch.device | None = None,
    ) -> list[int]:
        """Autoregressively generates tokens conditioned on [MAC_prefix, prompt_ids]."""
        dev = device or next(self.adapter.parameters()).device
        seq = list(prompt_ids)
        generated: list[int] = []

        prompt_tensor = torch.tensor([seq], dtype=torch.long, device=dev)
        prompt_embeds = self.embed_ids(prompt_tensor)
        prefix_embeds = self.adapter.retrieve_prefix(
            prompt_embeds, state, out_dtype=prompt_embeds.dtype
        )
        cur_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

        for _ in range(max_new_tokens):
            out = self.base_model(inputs_embeds=cur_embeds)
            logits = out.logits[:, -1, :]
            if temperature <= 1e-5:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled = logits / temperature
                probs = F.softmax(scaled, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = int(sorted_idx.gather(-1, sampled).item())
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_id)
            if next_id == eos_token_id:
                break
            next_emb = self.embed_ids(torch.tensor([[next_id]], dtype=torch.long, device=dev))
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

        return generated

    def save_adapter(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mem_cfg": asdict(self.mem_cfg),
            "adapter_state_dict": self.adapter.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_adapter(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        self.adapter.load_state_dict(payload["adapter_state_dict"])
        return payload.get("extra", {})


def load_base_llm_and_tokenizer(
    model_name: str,
    use_mock: bool = False,
    device: torch.device | None = None,
) -> tuple[nn.Module, Any, int]:
    """
    Loads either a real HuggingFace CausalLM (e.g. Qwen/Qwen2.5-0.5B-Instruct)
    or TinyMockCausalLM when use_mock=True.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_mock:
        tok = TinyMockTokenizer()
        model = TinyMockCausalLM(vocab_size=tok.vocab_size, hidden_size=128, num_layers=2).to(dev)
        return model, tok, 128

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(dev)
    model.eval()
    return model, tok, int(model.config.hidden_size)
