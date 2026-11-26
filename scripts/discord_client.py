"""Lightweight, zero-dependency Discord client for cloud / Remote Agent workflows.

Uses only the Python standard library (no pip install required) and reads credentials
from environment variables or a local `.env` file in the repository root.

Examples:
    # Send a notification via Webhook or Bot API
    python scripts/discord_client.py send "Training job finished!"

    # Read the latest 10 messages from a channel (requires DISCORD_BOT_TOKEN)
    python scripts/discord_client.py read --limit 10

    # Read messages as JSON (for autonomous agent parsing)
    python scripts/discord_client.py read --limit 5 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DISCORD_API_BASE = "https://discord.com/api/v10"


def load_dotenv(env_path: Path | None = None) -> None:
    """Load key=value pairs from .env without overriding existing env vars."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    req_headers = {"User-Agent": "TitanAgent-DiscordClient/1.0"}
    if headers:
        req_headers.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Discord API error {e.code}: {err_body}") from e


def send_message(
    content: str,
    channel_id: str | None = None,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    """Send a message using Bot API (if channel_id + token available) or Webhook."""
    load_dotenv()
    token = os.environ.get("DISCORD_BOT_TOKEN")
    resolved_channel = channel_id or os.environ.get("DISCORD_CHANNEL_ID")
    resolved_webhook = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")

    # Discord message limit is 2000 characters
    if len(content) > 2000:
        content = content[:1997] + "..."

    if token and resolved_channel and not webhook_url:
        url = f"{DISCORD_API_BASE}/channels/{resolved_channel}/messages"
        return _http_request(
            url,
            method="POST",
            headers={"Authorization": f"Bot {token}"},
            payload={"content": content},
        )

    if resolved_webhook:
        return _http_request(
            resolved_webhook,
            method="POST",
            payload={"content": content},
        )

    raise ValueError(
        "Set either (DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID) or DISCORD_WEBHOOK_URL."
    )


def read_messages(channel_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent messages from a Discord channel using DISCORD_BOT_TOKEN."""
    load_dotenv()
    token = os.environ.get("DISCORD_BOT_TOKEN")
    resolved_channel = channel_id or os.environ.get("DISCORD_CHANNEL_ID")
    if not token or not resolved_channel:
        raise ValueError("Reading messages requires DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID.")

    limit = max(1, min(100, limit))
    url = f"{DISCORD_API_BASE}/channels/{resolved_channel}/messages?limit={limit}"
    messages = _http_request(
        url,
        method="GET",
        headers={"Authorization": f"Bot {token}"},
    )
    return list(reversed(messages))  # oldest first


def main() -> int:
    parser = argparse.ArgumentParser(description="TitanAgent Discord CLI helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_send = sub.add_parser("send", help="Send a message to Discord")
    p_send.add_argument("message", help="Message text to send")
    p_send.add_argument("--channel", default=None, help="Override DISCORD_CHANNEL_ID")
    p_send.add_argument("--webhook", default=None, help="Override DISCORD_WEBHOOK_URL")

    p_read = sub.add_parser("read", help="Read recent messages from a channel")
    p_read.add_argument("--channel", default=None, help="Override DISCORD_CHANNEL_ID")
    p_read.add_argument("--limit", type=int, default=10, help="Number of messages (1-100)")
    p_read.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()
    try:
        if args.command == "send":
            res = send_message(args.message, channel_id=args.channel, webhook_url=args.webhook)
            print(f"Sent message successfully (id={res.get('id', 'webhook')}).")
        elif args.command == "read":
            msgs = read_messages(channel_id=args.channel, limit=args.limit)
            if args.json:
                print(json.dumps(msgs, ensure_ascii=False, indent=2))
            else:
                for m in msgs:
                    author = m.get("author", {}).get("username", "unknown")
                    ts = m.get("timestamp", "")[:19]
                    print(f"[{ts}] {author}: {m.get('content', '')}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
