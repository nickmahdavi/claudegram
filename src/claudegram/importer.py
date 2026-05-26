import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .identity import UserInfo

from .message import UTC, Forward, Message, Reply

_COMMAND_RE = re.compile(r"^\s*/\w+(@\w+)?(\s|$)")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImportResult:
    messages: list[Message]
    total: int = 0
    dropped_system: int = 0
    dropped_commands: int = 0
    dropped_service: int = 0
    dropped_non_text: int = 0
    # Identities recovered from the export, keyed by user_id. Telegram Desktop
    # exports carry a display name ("from") but no @handle, so username is a
    # placeholder. The bot seeds these into the store for users it doesn't
    # already know, so imported history renders with real names instead of the
    # "User <id>" fallback (live messages still take precedence; see note_user).
    users: dict[int, UserInfo] = field(default_factory=dict)

    @property
    def kept(self) -> int:
        return len(self.messages)


def _flatten_text(raw, entities) -> tuple[str, bool]:
    if isinstance(raw, list):
        text = "".join(
            frag if isinstance(frag, str) else frag.get("text", "") if isinstance(frag, dict) else ""
            for frag in raw
        )
    elif isinstance(raw, str):
        text = raw
    elif isinstance(entities, list) and entities:
        text = "".join(
            frag.get("text", "") if isinstance(frag, dict) else ""
            for frag in entities
        )
    else:
        text = ""

    is_command = bool(text) and bool(_COMMAND_RE.match(text))
    return text, is_command


def _parse_ts(unixtime_str: Optional[str], iso_fallback: Optional[str]) -> Optional[datetime]:
    if unixtime_str:
        try:
            return datetime.fromtimestamp(int(unixtime_str), tz=UTC)
        except (TypeError, ValueError):
            logger.debug("Bad date_unixtime %r; falling back to ISO date", unixtime_str)
    if iso_fallback:
        try:
            dt = datetime.fromisoformat(iso_fallback)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (TypeError, ValueError):
            logger.debug("Bad ISO date %r", iso_fallback)
    return None


def _extract_user_id(from_id) -> Optional[int]:
    if not isinstance(from_id, str):
        return None
    if not from_id.startswith("user"):
        return None
    try:
        return int(from_id[len("user"):])
    except ValueError:
        return None


def _photo_decorated(text: str, has_photo: bool) -> str:
    if not has_photo:
        return text
    if text:
        return f"[photo] {text}"
    return "[photo]"


def parse_export(
    path: Path,
    bot: UserInfo,
    system_prefix: str
) -> ImportResult:
    logger.info("Parsing export from %s (bot_user_id=%d)", path, bot.user_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("messages") or []
    result = ImportResult(messages=[])
    result.total = len(raw)
    by_id: dict[int, Message] = {}
    ordered: list[Message] = []

    for m in raw:
        if not isinstance(m, dict):
            result.dropped_non_text += 1
            continue
        if m.get("type") == "service":
            result.dropped_service += 1
            continue

        text, is_command = _flatten_text(m.get("text"), m.get("text_entities"))
        if is_command:
            result.dropped_commands += 1
            continue

        has_photo = bool(m.get("photo"))
        text = _photo_decorated(text, has_photo)
        if not text:
            result.dropped_non_text += 1
            continue

        if text.startswith(system_prefix):
            result.dropped_system += 1
            continue

        user_id = _extract_user_id(m.get("from_id"))
        if user_id is None:
            result.dropped_non_text += 1
            continue

        ts = _parse_ts(m.get("date_unixtime"), m.get("date"))
        if ts is None:
            result.dropped_non_text += 1
            continue

        msg_id = m.get("id")
        if not isinstance(msg_id, int):
            result.dropped_non_text += 1
            continue

        # Record identity only once the message has cleared every drop filter,
        # so result.users stays exactly the set of participants present in the
        # kept history (no orphans for users whose messages were all dropped).
        if user_id not in result.users:
            from_name = m.get("from")
            result.users[user_id] = UserInfo(
                user_id=user_id,
                username=f"user_{user_id}",
                display_name=from_name if isinstance(from_name, str) and from_name.strip() else f"user_{user_id}",
            )

        reply_to_raw = m.get("reply_to_message_id")
        reply_to = reply_to_raw if isinstance(reply_to_raw, int) else None

        # Telegram Desktop almost always serializes `forwarded_from` as a bare
        # display-name string (no @handle, no original send time). Channel-origin
        # posts occasionally appear as a dict with structured origin metadata; we
        # don't parse those and they're silently treated as "no forward." Worth
        # revisiting if it ever shows up at volume.
        forwarded_from = m.get("forwarded_from")
        forward = (
            Forward(display_name=forwarded_from, username="", ts=None, user_id=None)
            if isinstance(forwarded_from, str) and forwarded_from
            else None
        )

        msg = Message(
            id=msg_id,
            ts=ts,
            user_id=user_id,
            text=text,
            reply_to=reply_to,
            reply=None,
            forward=forward,
        )
        by_id[msg_id] = msg
        ordered.append(msg)

    for msg in ordered:
        if msg.reply_to is None:
            continue
        parent = by_id.get(msg.reply_to)
        if parent is None:
            continue
        msg.reply = Reply(
            user_id=parent.user_id,
            text=parent.text,
            is_quote=False,  # Telegram Desktop export doesn't preserve quote subsets
            ts=parent.ts,
        )

    result.messages = ordered
    logger.info(
        "Parsed export %s: kept=%d total=%d users=%d dropped_system=%d dropped_commands=%d "
        "dropped_service=%d dropped_non_text=%d",
        path, result.kept, result.total, len(result.users),
        result.dropped_system, result.dropped_commands,
        result.dropped_service, result.dropped_non_text,
    )
    return result
