import collections
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from itertools import groupby
from pathlib import Path
from typing import ClassVar, Iterable, Literal, Optional, Self, Union, cast

from anthropic.types import MessageParam

logger = logging.getLogger(__name__)

type PromptMode = Literal["chat", "prefill"]

PathLike = Union[str, Path]

UTC = timezone.utc

SYSTEM_PREFIX = "<System>"


def fmt_offset(tz: tzinfo, at: Optional[datetime] = None) -> str:
    moment = at or datetime.now(UTC)
    off = moment.astimezone(tz).strftime("%z")  # e.g. '-0400', '+0530'
    if not off:
        return "+00"
    sign, hh, mm = off[0], off[1:3], off[3:5]
    if mm == "00":
        return f"{sign}{hh}"
    return f"{sign}{hh}{mm}"


def format_tag(display_name: str, username: str, ts: Optional[datetime], tz: tzinfo = UTC) -> str:
    if ts is None:
        return f"<{display_name} (@{username}) : ??:?? {fmt_offset(tz)}>"
    local = ts.astimezone(tz)
    return f"<{display_name} (@{username}) : {local.strftime('%H:%M')} {fmt_offset(tz, ts)}>"


@dataclass(slots=True)
class Reply:
    display_name: str
    username: str
    text: str
    is_quote: bool
    ts: Optional[datetime]
    user_id: Optional[int] = None
    QUOTE_CHAR_LIMIT: ClassVar[int] = 30

    def render(self, tz: tzinfo = UTC) -> str:
        prefix = ">" if self.is_quote else "re."
        quote_mark = '"' if self.is_quote else ""
        truncated = self.text[:self.QUOTE_CHAR_LIMIT]
        ellipsis = "..." if len(self.text) > self.QUOTE_CHAR_LIMIT else ""
        identity = format_tag(self.display_name, self.username, self.ts, tz)
        return f"{prefix} {identity} {quote_mark}{truncated}{quote_mark}{ellipsis}"

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "username": self.username,
            "text": self.text,
            "is_quote": self.is_quote,
            "ts": self.ts.isoformat() if self.ts else None,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        ts_raw = d.get("ts")
        return cls(
            display_name=d.get("display_name", ""),
            username=d.get("username", ""),
            text=d.get("text", ""),
            is_quote=d.get("is_quote", False),
            ts=datetime.fromisoformat(ts_raw) if ts_raw else None,
            user_id=d.get("user_id"),
        )


@dataclass(slots=True)
class Message:
    id: int
    ts: datetime
    username: str
    display_name: str
    text: str
    reply_to: Optional[int] = None
    reply: Optional[Reply] = None
    user_id: Optional[int] = None
    _tokens: int | None = field(init=False, repr=False, default=None)
    _message: str | None = field(init=False, repr=False, default=None)

    @property
    def tokens(self) -> int:
        if self._tokens is None:
            self._tokens = len(self.message) // 4 + 5
        return self._tokens

    @property
    def message(self) -> str:
        if self._message is None:
            self._message = self.render(UTC)
        return self._message

    def render(self, tz: tzinfo = UTC) -> str:
        identity = format_tag(self.display_name, self.username, self.ts, tz)
        body = f"{identity} {self.text}"
        if self.reply is not None:
            return f"{self.reply.render(tz)}\n{body}"
        return body

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat(),
            "username": self.username,
            "display_name": self.display_name,
            "text": self.text,
            "reply_to": self.reply_to,
            "reply": self.reply.to_dict() if self.reply else None,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            username=d["username"],
            display_name=d["display_name"],
            text=d["text"],
            reply_to=d.get("reply_to"),
            reply=Reply.from_dict(d["reply"]) if d.get("reply") else None,
            user_id=d.get("user_id"),
        )


class Window:
    def __init__(self, budget: int):
        self.budget = budget
        self._history: collections.deque[Message] = collections.deque()
        self._full: list[Message] = []
        self._total = 0
        self._persisted = 0

    def append(self, message: Message) -> list[Message]:
        self._full.append(message)
        self._history.append(message)
        self._total += message.tokens
        evicted = []
        while self._total > self.budget and self._history:
            removed = self._history.popleft()
            self._total -= removed.tokens
            evicted.append(removed)
        return evicted

    def __iter__(self) -> Iterable[Message]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def tokens(self) -> int:
        return self._total
    
    @property
    def size(self) -> int:
        return len(self._full)

    def snapshot(self) -> list[Message]:
        return list(self._history)

    def history(self, model_username: str, mode: PromptMode, display_tz: tzinfo = UTC) -> list[MessageParam]:
        return render_history(self.snapshot(), model_username, mode, display_tz)

    def write(self, path: PathLike) -> int:
        new = self._full[self._persisted:]
        if not new:
            return 0
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for msg in new:
                f.write(json.dumps(msg.to_dict()) + "\n")
        self._persisted = len(self._full)
        logger.debug("Wrote %d message(s) to %s", len(new), p)
        return len(new)

    @classmethod
    def from_file(cls, path: PathLike, budget: int) -> Self:
        window = cls(budget=budget)
        p = Path(path)
        if not p.exists():
            return window
        messages: list[Message] = []
        skipped = 0
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON in %s line %d: %s", p, lineno, e)
                    skipped += 1
                    continue
                if lineno == 1 and isinstance(obj, dict) and "id" not in obj:
                    logger.debug("Schema header in %s: %s", p, obj)
                    continue
                try:
                    messages.append(Message.from_dict(obj))
                except Exception as e:
                    logger.warning("Skipping unreadable message in %s line %d: %s", p, lineno, e)
                    skipped += 1
        if skipped > 0:
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg.to_dict()) + "\n")
            tmp.replace(p)
            logger.info("Rewrote %s after skipping %d malformed line(s)", p, skipped)
        window._load(messages, all_persisted=True)
        logger.debug(
            "Loaded %d message(s) from %s (working set: %d, tokens: %d, skipped: %d)",
            len(messages), p, len(window), window.tokens, skipped,
        )
        return window

    def _load(self, messages: list[Message], *, all_persisted: bool) -> None:
        self._full = list(messages)
        self._persisted = len(self._full) if all_persisted else 0
        for msg in reversed(messages):
            if self._total + msg.tokens > self.budget:
                break
            self._history.appendleft(msg)
            self._total += msg.tokens


def render_history(
    messages: list[Message],
    model_username: str,
    mode: PromptMode,
    display_tz: tzinfo = UTC,
) -> list[MessageParam]:
    if mode == "chat":
        hist = list(messages)
        while hist and hist[0].username == model_username:
            hist.pop(0)

        rendered: list[tuple[Literal["user", "assistant"], str]] = []
        last_emitted_date = None
        for m in hist:
            is_assistant = m.username == model_username
            body = m.text if is_assistant else m.render(display_tz)
            d = m.ts.astimezone(display_tz).date() if m.ts else None
            if not is_assistant and d is not None and d != last_emitted_date:
                body = f"--- {d.isoformat()} ---\n{body}"
                last_emitted_date = d
            rendered.append(("assistant" if is_assistant else "user", body))

        result: list[MessageParam] = []
        for role, group in groupby(rendered, key=lambda x: x[0]):
            chunks = [text for _, text in group]
            content = " ".join(chunks) if role == "assistant" else "\n\n".join(chunks)
            result.append(MessageParam(role=cast(Literal["user", "assistant"], role), content=content))
        return result
    if mode == "prefill":
        # TODO: frag/stop handling (not blocking, we use 4.7)
        chats = [msg.message for msg in messages]
        # usually had CLI sim prompt + "cat untitled.txt"
        return [
            MessageParam(role="user", content="."),
            MessageParam(role="assistant", content="\n\n".join(chats)),
        ]
    raise ValueError(f"invalid mode {mode}")
