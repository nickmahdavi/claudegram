import collections
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Iterable, Optional, Self, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

UTC = ZoneInfo("UTC")

@dataclass(slots=True)
class Reply:
    user_id: int
    text: str
    is_quote: bool
    ts: datetime
    QUOTE_CHAR_LIMIT: ClassVar[int] = 30

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "text": self.text,
            "is_quote": self.is_quote,
            "ts": self.ts.isoformat()
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            user_id=int(d["user_id"]),
            text=d["text"],
            is_quote=d["is_quote"],
            ts=datetime.fromisoformat(d["ts"])
        )

@dataclass(slots=True)
class Message:
    id: int
    ts: datetime
    user_id: int
    text: str
    reply_to: Optional[int] = None
    reply: Optional[Reply] = None
    _tokens: int | None = field(init=False, repr=False, default=None)

    TAG_OVERHEAD: ClassVar[int] = 12
    REPLY_OVERHEAD: ClassVar[int] = 20

    @property
    def tokens(self) -> int:
        if self._tokens is None:
            n = len(self.text) // 4 + self.TAG_OVERHEAD
            if self.reply is not None:
                n += self.REPLY_OVERHEAD
            self._tokens = n
        return self._tokens

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat(),
            "user_id": self.user_id,
            "text": self.text,
            "reply_to": self.reply_to,
            "reply": self.reply.to_dict() if self.reply else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        reply = None
        raw = d.get("reply")
        if raw is not None:
            try:
                reply = Reply.from_dict(raw)
            except Exception as e:
                logger.warning("Failed to parse reply in message %s: %s", d.get("id", "with missing id"), e)
        return cls(
            id=int(d["id"]),
            ts=datetime.fromisoformat(d["ts"]),
            user_id=int(d["user_id"]),
            text=d["text"],
            reply_to=d.get("reply_to"),
            reply=reply,
        )


class Window:
    # When over budget, evict down to EVICT_TARGET * budget instead of "just under budget."
    # This gives a stable prefix for many subsequent appends, which Anthropic's prompt cache
    # requires to hit. See the cache_control markers in model.complete().
    EVICT_TARGET: ClassVar[float] = 0.7

    def __init__(self, budget: int):
        self.budget = budget
        self._history: collections.deque[Message] = collections.deque()
        self._full: list[Message] = []
        self._total = 0
        self._persisted = 0
        self._participants = set()

    def append(self, message: Message) -> list[Message]:
        self._full.append(message)
        self._history.append(message)
        self._total += message.tokens
        self._participants.add(message.user_id)
        if self._total <= self.budget:
            return []
        target = int(self.budget * self.EVICT_TARGET)
        evicted: list[Message] = []
        # Keep at least one message so we never ship messages=[] to the API
        # (an oversized lone message still goes out as-is)
        while self._total > target and len(self._history) > 1:
            removed = self._history.popleft()
            self._total -= removed.tokens
            evicted.append(removed)
        return evicted

    # realizing we have little to no security elsewhere...
    # but also not sure what the threat model is here, so. Caveat operator.
    def known_users(self) -> set[int]:
        return self._participants.copy()

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
            logger.info("Skipped %d malformed line(s)", skipped)
        window._load(messages, all_persisted=True)
        logger.debug(
            "Loaded %d message(s) from %s (working set: %d, tokens: %d, skipped: %d)",
            len(messages), p, len(window), window.tokens, skipped,
        )
        return window

    def _load(self, messages: list[Message], *, all_persisted: bool) -> None:
        self._full = list(messages)
        self._persisted = len(self._full) if all_persisted else 0
        for message in messages:
            self._participants.add(message.user_id)
        # Load to the eviction target, not full budget, or otherwise the working set comes back at
        # ~budget and the first append after a restart immediately evicts.
        target = int(self.budget * self.EVICT_TARGET)
        for msg in reversed(messages):
            if self._total + msg.tokens > target:
                break
            self._history.appendleft(msg)
            self._total += msg.tokens
