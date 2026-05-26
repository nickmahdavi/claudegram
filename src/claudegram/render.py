from datetime import date, datetime
from enum import StrEnum
from itertools import groupby
from typing import Callable, Literal, Optional, cast
from zoneinfo import ZoneInfo

from anthropic.types import MessageParam

from .identity import UserInfo
from .message import UTC, Forward, Message, Reply


Resolver = Callable[[int], UserInfo]

class RenderMode(StrEnum):
    """How a list of Messages is shaped into SDK message turns.

    CHAT alternates user/assistant turns (the normal case).
    PREFILL uses the assistant-prefill trick: one user "." followed by an
    assistant turn pre-filled with the rendered conversation.

    Independent of which system prompt is selected; see model.PromptMode for
    that. Multiple PromptModes (CHAT, CHAT_PRIVATE, ...) may share a single
    RenderMode.
    """
    CHAT = "chat"
    PREFILL = "prefill"

def build_tz_directory(users: set[int], resolve: Resolver) -> str:
    lines = ["User timezone directory (UTC offsets in message tags; convert as needed):"]
    for user_id in sorted(users):
        user = resolve(user_id)
        user_tz_str = user.tz.key if user.tz else 'unset (00?), treat as UTC'
        lines.append(f"  @{user.username} (id={user.user_id}) — {user_tz_str}")

    return "\n".join(lines)

def fmt_offset(tz: Optional[ZoneInfo], at: Optional[datetime] = None) -> str:
    if not tz:
        return "+0000?"
    moment = at or datetime.now(UTC)
    off = moment.astimezone(tz).strftime("%z")  # e.g. '-0400', '+0530'
    sign, hh, mm = off[0], off[1:3], off[3:5]
    if mm == "00":
        return f"{sign}{hh}"
    return f"{sign}{hh}{mm}"

def format_tag(username: str, display_name: str, ts: Optional[datetime], display_tz: Optional[ZoneInfo] = None) -> str:
    # `username` may be falsy (forwards from hidden users / channels without a
    # public @handle) -- drop the `(@handle)` segment rather than emit `(@)`.
    # `ts` may be missing for forwards imported from Telegram Desktop (bare name,
    # no timestamp) -- drop the time fragment rather than emit a fake-precision
    # `??:?? +0000?` placeholder. Chat participants always have both, so only
    # forwards hit these branches.
    # `display_name` should always be set, but guard against an empty/whitespace
    # one (e.g. a corrupt persisted forward) so we never render bare `<>`.
    display_name = display_name.strip() if display_name and display_name.strip() else "unknown"
    handle = f" (@{username})" if username else ""
    if ts is None:
        return f"<{display_name}{handle}>"
    local = ts.astimezone(display_tz or UTC)
    return f"<{display_name}{handle} : {local.strftime('%H:%M')} {fmt_offset(display_tz, ts)}>"

def render_forward(forward: Forward, display_tz: Optional[ZoneInfo] = None) -> str:
    # Rendered from the Forward's own snapshot (original author at original time),
    # not via the resolver -- the origin usually isn't a chat participant.
    tag = format_tag(forward.username, forward.display_name, forward.ts, display_tz)
    return f"fwd. {tag}"

def render_quote(reply: Reply, resolve: Resolver, display_tz: Optional[ZoneInfo] = None) -> str:
    user_info = resolve(reply.user_id)
    prefix = ">" if reply.is_quote else "re."
    quote_mark = '"' if reply.is_quote else ""
    truncated = reply.text[:reply.QUOTE_CHAR_LIMIT]
    ellipsis = "..." if len(reply.text) > reply.QUOTE_CHAR_LIMIT else ""
    tag = format_tag(user_info.username, user_info.display_name, reply.ts, display_tz)
    return f"{prefix} {tag} {quote_mark}{truncated}{quote_mark}{ellipsis}"

def render_message(message: Message, resolve: Resolver, display_tz: Optional[ZoneInfo] = None) -> str:
    # display_tz should always be passed in a group chat.
    #  (Right now, it's UTC; in the future it might vary per-user.
    # If display_tz is set, all timestamps are rendered in that tz.
    #  Otherwise, we fall back to per-user timestamps.
    # If a user has no tz, we fall back to UTC and mark it with a "?" in the offset
    #  (e.g. "+0000?") to indicate that it might be wrong. Otherwise, we just use their tz.
    #  This branch usually only happens in private chats, but in the future we might want to
    #  allow it in groups as well (if it doesn't confuse models too much).
    user_info = resolve(message.user_id)
    tz = display_tz or user_info.tz
    tag = format_tag(user_info.username, user_info.display_name, message.ts, tz)
    lines = []
    if message.forward is not None:
        lines.append(render_forward(message.forward, tz))
    if message.reply is not None:
        lines.append(render_quote(message.reply, resolve, tz))
    lines.append(f"{tag} {message.text}")
    return "\n".join(lines)

def get_datemark(message: Message, resolve: Resolver, last_emitted_date: Optional[date], display_tz: Optional[ZoneInfo] = None) -> tuple[str, date]:
    # We also render a date separator if the message is on a different day from the last emitted
    #  message. The logic is identical to the display_tz logic. Note that date separators might
    #  appear out of chronological order if messages from different users with different tz's are
    #  interleaved, which isn't unambiguously bad.
    # Returns (message, new_last_emitted_date)
    user_info = resolve(message.user_id)
    message_date = message.ts.astimezone(display_tz or user_info.tz or UTC).date()
    if last_emitted_date is None or message_date != last_emitted_date:
        return (f"--- {message_date.isoformat()} ---\n\n", message_date)
    return ("", last_emitted_date)

def render_history(
    messages: list[Message],
    bot_info: UserInfo,
    mode: RenderMode,
    resolve: Resolver,
    display_tz: Optional[ZoneInfo] = None
) -> list[MessageParam]:
    if mode == RenderMode.CHAT:
        hist = list(messages)
        while hist and hist[0].user_id == bot_info.user_id:
            hist.pop(0)

        rendered: list[tuple[Literal["user", "assistant"], str]] = []
        last_emitted_date = None
        for msg in hist:
            is_assistant = msg.user_id == bot_info.user_id
            body = msg.text if is_assistant else render_message(msg, resolve, display_tz)
            if not is_assistant:
                datemark, last_emitted_date = get_datemark(msg, resolve, last_emitted_date, display_tz)
                body = f"{datemark}{body}"
            rendered.append(("assistant" if is_assistant else "user", body))

        result: list[MessageParam] = []
        for role, group in groupby(rendered, key=lambda x: x[0]):
            chunks = [text for _, text in group]
            content = "".join(chunks) if role == "assistant" else "\n\n".join(chunks)
            result.append(MessageParam(role=cast(Literal["user", "assistant"], role), content=content))
        return result

    if mode == RenderMode.PREFILL:
        # TODO: frag/stop handling (not blocking, we use 4.7)
        # usually had CLI sim prompt + "cat untitled.txt"
        # Unimplemented
        raise NotImplementedError("Prefill mode not supported")

    raise ValueError(f"invalid mode {mode}")
