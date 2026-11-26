"""
Phase 2: 1-Stream Lifelong Japanese Chat UI (Gradio Web UI + CLI)
backed by Frozen Japanese LLM + Titans (MAC) Neural Long-Term Memory.

Key Features:
  - Single lifelong conversation stream (no "new chat" by default).
  - Applies Exp 3.5 findings:
      * Turn-level <think> premise rehearsal for stable memory retention.
      * Old <think> blocks in the active short-term window are pruned first.
      * Turns exceeding `max_active_turns` are compressed into Titans Neural
        Memory fast weights M_t (test-time surprise gradient + forget gate)
        and evicted from the short-term window.
  - Persistent Memory State: Save/Load M_t + conversation log (.pt) across restarts.
  - Live Memory Inspector in Gradio: shows active vs archived turns, surprise,
    forget gate alpha, and memory update count.

Usage:
  # Local CPU mock test (CLI):
  python phase2_llm_titans_chat/chat_app.py --mock --cli

  # Full Japanese LLM + Gradio UI (GPU recommended):
  python phase2_llm_titans_chat/chat_app.py --model_name Qwen/Qwen2.5-0.5B-Instruct --adapter_ckpt checkpoints/phase2_mac_adapter/titans_mac_adapter.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch

from titans_mac_adapter import (
    TitansMACLLM,
    TitansMemoryConfig,
    TitansMemoryState,
    load_base_llm_and_tokenizer,
)


THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

SYSTEM_PROMPT_JA = (
    "あなたは1本道の長期会話でユーザーに寄り添って育つパーソナルAIエージェントです。"
    "応答の前に必要に応じて<think>記憶リハーサル: ...</think>で重要な前提や変更点を簡潔に確認し、"
    "自然な日本語で丁寧に答えてください。"
)


@dataclass
class ChatTurn:
    user: str
    assistant: str
    surprise: float = 0.0
    forget_rate: float = 0.0


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


class OneStreamTitanSession:
    """Manages a single continuous conversation + Titans Neural Memory fast weights."""

    def __init__(
        self,
        wrapper: TitansMACLLM,
        tokenizer: Any,
        device: torch.device,
        max_active_turns: int = 3,
    ) -> None:
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.device = device
        self.max_active_turns = max_active_turns

        self.memory_state: TitansMemoryState = wrapper.init_memory_state(batch_size=1, device=device)
        self.active_turns: list[ChatTurn] = []
        self.full_history: list[ChatTurn] = []
        self.archived_turn_count: int = 0

    def _format_prompt(self, new_user_msg: str) -> str:
        lines = [f"システム: {SYSTEM_PROMPT_JA}"]
        # Exp 3.5 policy: keep recent active turns, pruning <think> from older active turns
        for i, t in enumerate(self.active_turns):
            is_oldest_active = i < len(self.active_turns) - 1
            asst_text = strip_think(t.assistant) if is_oldest_active else t.assistant
            lines.append(f"ユーザー: {t.user}")
            lines.append(f"アシスタント: {asst_text}")
        lines.append(f"ユーザー: {new_user_msg}")
        lines.append("アシスタント: ")
        return "\n".join(lines)

    @torch.no_grad()
    def send_message(
        self,
        user_msg: str,
        max_new_tokens: int = 160,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> tuple[str, dict[str, Any]]:
        self.wrapper.adapter.eval()
        prompt_text = self._format_prompt(user_msg)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        gen_ids = self.wrapper.generate_with_memory(
            prompt_ids=prompt_ids,
            state=self.memory_state,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=self.device,
        )
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Memorize this turn into Titans Neural Memory fast weights M_t (test-time update)
        turn_segment = f"ユーザー: {user_msg}\nアシスタント: {reply}"
        seg_ids = torch.tensor(
            [self.tokenizer.encode(turn_segment, add_special_tokens=False)],
            dtype=torch.long,
            device=self.device,
        )
        self.memory_state, stats = self.wrapper.memorize_segment(seg_ids, self.memory_state)
        self.memory_state.detach_()

        turn = ChatTurn(
            user=user_msg,
            assistant=reply,
            surprise=stats["surprise"],
            forget_rate=stats["forget_rate"],
        )
        self.full_history.append(turn)
        self.active_turns.append(turn)

        # Evict oldest turn from short-term window once limit is exceeded
        # (its information now lives exclusively in Titans fast weights M_t)
        while len(self.active_turns) > self.max_active_turns:
            self.active_turns.pop(0)
            self.archived_turn_count += 1

        return reply, self.get_monitor_dict()

    def get_monitor_dict(self) -> dict[str, Any]:
        return {
            "total_turns": len(self.full_history),
            "active_window_turns": len(self.active_turns),
            "archived_to_titans_turns": self.archived_turn_count,
            "titans_memory_updates": self.memory_state.num_updates,
            "last_surprise": round(self.memory_state.last_surprise, 5),
            "last_forget_rate": round(self.memory_state.last_forget_rate, 5),
        }

    def format_monitor_markdown(self) -> str:
        m = self.get_monitor_dict()
        return (
            "### Titans 長期記憶モニター\n"
            f"- **総ターン数（1本道）**: `{m['total_turns']}`\n"
            f"- **短期ウィンドウ内**: `{m['active_window_turns']}` / `{self.max_active_turns}` ターン\n"
            f"- **Titans 記憶へ圧縮済み**: `{m['archived_to_titans_turns']}` ターン\n"
            f"- **Neural Memory 更新回数**: `{m['titans_memory_updates']}`\n"
            f"- **直近 Surprise（驚き度）**: `{m['last_surprise']}`\n"
            f"- **直近 忘却ゲート (α)**: `{m['last_forget_rate']}`\n"
        )

    def save_session(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory_state": self.memory_state.to_dict(),
            "active_turns": [t.__dict__ for t in self.active_turns],
            "full_history": [t.__dict__ for t in self.full_history],
            "archived_turn_count": self.archived_turn_count,
            "max_active_turns": self.max_active_turns,
        }
        torch.save(payload, path)
        return f"記憶状態を保存しました: {path}"

    def load_session(self, path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            return f"ファイルが見つかりません: {path}"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_state = TitansMemoryState.from_dict(payload["memory_state"], self.device)
        self.active_turns = [ChatTurn(**d) for d in payload["active_turns"]]
        self.full_history = [ChatTurn(**d) for d in payload["full_history"]]
        self.archived_turn_count = int(payload.get("archived_turn_count", 0))
        return f"記憶状態を復元しました（総ターン数: {len(self.full_history)}）"

    def reset_for_testing(self) -> str:
        self.memory_state = self.wrapper.init_memory_state(batch_size=1, device=self.device)
        self.active_turns.clear()
        self.full_history.clear()
        self.archived_turn_count = 0
        return "検証用に記憶と1本道履歴をリセットしました。"


def launch_gradio(session: OneStreamTitanSession, session_file: str, share: bool = False) -> None:
    import gradio as gr

    def build_chatbot_messages(show_think: bool) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        for t in session.full_history:
            msgs.append({"role": "user", "content": t.user})
            asst = t.assistant if show_think else (strip_think(t.assistant) or t.assistant)
            msgs.append({"role": "assistant", "content": asst})
        return msgs

    def on_send(user_msg: str, show_think: bool, temperature: float, max_new_tokens: int):
        if not user_msg.strip():
            return "", build_chatbot_messages(show_think), session.format_monitor_markdown()
        session.send_message(
            user_msg.strip(),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
        )
        return "", build_chatbot_messages(show_think), session.format_monitor_markdown()

    def on_save(path_str: str):
        return session.save_session(path_str), session.format_monitor_markdown()

    def on_load(path_str: str, show_think: bool):
        msg = session.load_session(path_str)
        return msg, build_chatbot_messages(show_think), session.format_monitor_markdown()

    def on_reset(show_think: bool):
        msg = session.reset_for_testing()
        return msg, build_chatbot_messages(show_think), session.format_monitor_markdown()

    with gr.Blocks(title="TitanAgent Phase 2 — 1本道日本語チャット") as demo:
        gr.Markdown("# TitanAgent Phase 2: 1本道・超長期記憶チャット（Titans MAC）")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="1本道会話ストリーム", height=460, type="messages")
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="メッセージを入力（例: はじめまして、〇〇です。今の目標は...）",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("送信", variant="primary", scale=1)
            with gr.Column(scale=1):
                monitor_md = gr.Markdown(session.format_monitor_markdown())
                show_think = gr.Checkbox(value=True, label="<think>（記憶リハーサル）を表示")
                temperature = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Temperature")
                max_new_tokens = gr.Slider(32, 320, value=160, step=16, label="最大生成トークン")
                gr.Markdown("---")
                state_path = gr.Textbox(value=session_file, label="記憶状態ファイル (.pt)")
                with gr.Row():
                    save_btn = gr.Button("記憶を保存")
                    load_btn = gr.Button("記憶を復元")
                reset_btn = gr.Button("検証用リセット", variant="stop")
                status_out = gr.Textbox(label="ステータス", interactive=False)

        send_btn.click(
            on_send,
            inputs=[msg_box, show_think, temperature, max_new_tokens],
            outputs=[msg_box, chatbot, monitor_md],
        )
        msg_box.submit(
            on_send,
            inputs=[msg_box, show_think, temperature, max_new_tokens],
            outputs=[msg_box, chatbot, monitor_md],
        )
        save_btn.click(on_save, inputs=[state_path], outputs=[status_out, monitor_md])
        load_btn.click(on_load, inputs=[state_path, show_think], outputs=[status_out, chatbot, monitor_md])
        reset_btn.click(on_reset, inputs=[show_think], outputs=[status_out, chatbot, monitor_md])

    demo.launch(share=share)


def run_cli(session: OneStreamTitanSession, session_file: str) -> None:
    print("=== TitanAgent Phase 2 (CLI 1本道チャット) ===")
    print("コマンド: /save, /load, /status, /quit")
    while True:
        try:
            user_msg = input("\nあなた> ").strip()
        except EOFError:
            break
        if not user_msg:
            continue
        if user_msg == "/quit":
            break
        if user_msg == "/save":
            print(session.save_session(session_file))
            continue
        if user_msg == "/load":
            print(session.load_session(session_file))
            continue
        if user_msg == "/status":
            print(session.get_monitor_dict())
            continue
        reply, stats = session.send_message(user_msg, temperature=0.4)
        print(f"AI> {reply}")
        print(f"   [Titans] updates={stats['titans_memory_updates']} archived={stats['archived_to_titans_turns']} surprise={stats['last_surprise']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TitanAgent Phase 2 1-Stream Japanese Chat")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_ckpt", type=str, default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max_active_turns", type=int, default=3)
    parser.add_argument("--session_file", type=str, default="checkpoints/phase2_session_state.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer, hidden_size = load_base_llm_and_tokenizer(
        args.model_name, use_mock=args.mock, device=device
    )

    if args.adapter_ckpt and Path(args.adapter_ckpt).exists():
        payload = torch.load(args.adapter_ckpt, map_location=device, weights_only=False)
        mem_cfg = TitansMemoryConfig(**payload["mem_cfg"])
        wrapper = TitansMACLLM(base_model, mem_cfg, freeze_base=True).to(device)
        wrapper.adapter.load_state_dict(payload["adapter_state_dict"])
        print(f"Loaded trained MAC adapter from {args.adapter_ckpt}")
    else:
        mem_cfg = TitansMemoryConfig(
            hidden_size=hidden_size,
            mem_dim=min(256, hidden_size) if args.mock else 256,
        )
        wrapper = TitansMACLLM(base_model, mem_cfg, freeze_base=True).to(device)
        print("Initialized fresh MAC adapter (no checkpoint loaded).")

    session = OneStreamTitanSession(
        wrapper=wrapper,
        tokenizer=tokenizer,
        device=device,
        max_active_turns=args.max_active_turns,
    )

    if args.cli:
        run_cli(session, args.session_file)
    else:
        try:
            launch_gradio(session, args.session_file, share=args.share)
        except ImportError:
            print("gradio が未インストールのため CLI モードで起動します（pip install gradio で Web UI 利用可能）。")
            run_cli(session, args.session_file)


if __name__ == "__main__":
    main()
