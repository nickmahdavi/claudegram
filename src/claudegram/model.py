import logging
from datetime import tzinfo
from enum import Enum
from typing import Callable, Optional
from string import Template
from zoneinfo import ZoneInfo

import anthropic
from anthropic.types import ModelParam, TextBlock

from .message import UTC, Message, PromptMode, fmt_offset, render_history

logger = logging.getLogger(__name__)


class ErrorClass(Enum):
    """Buckets for Anthropic API failures, used by the bot's error handling to
    decide what to say to the user and the admin, and whether to back off.

    TRANSIENT covers things worth retrying (rate limit, 5xx, connection drop).
    The anthropic SDK already retries these internally a couple times, so by
    the time we see them they're persistent enough to surface, just not
    necessarily for-real-broken.
    """
    TRANSIENT = "transient"
    CREDIT = "credit"
    AUTH = "auth"
    MODEL_NOT_FOUND = "model_not_found"
    BAD_REQUEST = "bad_request"
    UNKNOWN = "unknown"


def classify_error(exc: BaseException) -> tuple[ErrorClass, str]:
    """Map an exception (typically from `anthropic.Client.messages.create`)
    to a (class, short_human_description) pair. Description is one short
    phrase suitable for logs and admin DMs, no period.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return ErrorClass.TRANSIENT, "rate limited (429)"
    if isinstance(exc, anthropic.InternalServerError):
        return ErrorClass.TRANSIENT, "anthropic 5xx"
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorClass.TRANSIENT, "connection failure"
    if isinstance(exc, anthropic.AuthenticationError):
        return ErrorClass.AUTH, "auth failure (bad/expired key)"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ErrorClass.AUTH, "permission denied"
    if isinstance(exc, anthropic.NotFoundError):
        return ErrorClass.MODEL_NOT_FOUND, "model not found"
    if isinstance(exc, anthropic.BadRequestError):
        # Anthropic surfaces "out of credit" / "billing" as 400 BadRequest with
        # a typed message; sniff the text since the SDK doesnt give us a
        # dedicated exception class for it.
        msg = str(exc).lower()
        if any(k in msg for k in ("credit balance", "billing", "quota", "your credit")):
            return ErrorClass.CREDIT, "credit balance exhausted"
        return ErrorClass.BAD_REQUEST, f"bad request: {str(exc)[:120]}"
    if isinstance(exc, anthropic.UnprocessableEntityError):
        return ErrorClass.BAD_REQUEST, "unprocessable entity"
    if isinstance(exc, anthropic.APIError):
        # Catch-all for any other anthropic.* error we havent enumerated.
        return ErrorClass.UNKNOWN, f"anthropic api error: {type(exc).__name__}"
    return ErrorClass.UNKNOWN, type(exc).__name__


class Claude:
    NO_PREFILL = ("claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")

    KNOWN_MODELS = frozenset({
        "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1", "claude-opus-4-0",
        "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4-0",
        "claude-haiku-4-5",
    })

    MODEL_ALIASES: dict[str, str] = {
        "op4.7": "claude-opus-4-7",
        "op4.6": "claude-opus-4-6",
        "op4.5": "claude-opus-4-5",
        "op4.0": "claude-opus-4-1",
        "op4": "claude-opus-4-0",
        "s4.6": "claude-sonnet-4-6",
        "s4.5": "claude-sonnet-4-5",
        "s4.0": "claude-sonnet-4-0",
        "s4": "claude-sonnet-4-0",
        "h4.5": "claude-haiku-4-5",
    }

    SYSTEM_PROMPTS = {
        "prefill": "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with outputs of the commands.",
        "chat": "You're an LLM in a group conversation. Messages from other participants are prefixed with their name + handle + UTC time + offset suffix (e.g. '14:32 +00'). You should just send your messages like normal (no prefix).\nYour display name is $display_name, and your username is $username. Users might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name.\n$user_tz_directory\nHave fun!",
        "chat_private": "You're an LLM in a private (1:1) conversation with $partner_display_name (@$partner_username). Their messages appear in human / user turns prefixed with their name + handle + local time + offset suffix (e.g. '14:32 -04'). Timestamps are rendered in their timezone ($partner_tz). You should just send your messages like normal (no prefix).\nYour display name is $display_name, and your username is $username. They might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name. Have fun!",
    }

    def __init__(self, api_key: str, model: ModelParam, max_tokens: int):
        self.client = anthropic.Client(api_key=api_key)
        self._max_tokens = max_tokens
        self._model = model # default
        logger.info("Claude initialized: default_model=%s max_tokens=%d", model, max_tokens)

    def __repr__(self) -> str:
        return f"Claude(default_model={self._model!r}, max_tokens={self._max_tokens})"

    @property
    def model(self) -> str:
        return str(self._model)

    @classmethod
    def mode_for(cls, model: str) -> PromptMode:
        # Prefill mode is disabled until stop seqence detection is fixed
        return "chat"
        # return "chat" if model in cls.NO_PREFILL else "prefill"

    @staticmethod
    def _build_tz_directory(
        messages: list[Message],
        tz_lookup: Optional[Callable[[int], Optional[str]]],
    ) -> tuple[str, int]:
        if tz_lookup is None:
            return "", 0

        seen: dict[int, tuple[str, str]] = {}  # user_id -> (handle, display_name)
        for m in messages:
            if m.user_id is None:
                continue
            if m.user_id in seen:
                continue
            seen[m.user_id] = (m.username, m.display_name)

        if not seen:
            return "", 0

        known: list[tuple[int, str, str, str]] = []  # (user_id, handle, tz_name, offset)
        unknown: list[tuple[int, str]] = []  # (user_id, handle)
        for uid in sorted(seen):
            handle, _ = seen[uid]
            tz_name = tz_lookup(uid)
            if tz_name:
                try:
                    known.append((uid, handle, tz_name, fmt_offset(ZoneInfo(tz_name))))
                except Exception:
                    unknown.append((uid, handle))
            else:
                unknown.append((uid, handle))

        lines = ["User timezone directory (UTC offsets in message tags; convert as needed):"]
        for uid, handle, tz_name, off in known:
            lines.append(f"  @{handle} (id={uid}) — {tz_name} ({off})")
        for uid, handle in unknown:
            lines.append(f"  @{handle} (id={uid}) — unset (00?), treat as UTC")
        return "\n".join(lines), len(known) + len(unknown)

    def complete(
        self,
        messages: list[Message],
        window_tokens: int,
        username: str,
        display_name: str,
        model: Optional[ModelParam] = None,
        max_tokens: Optional[int] = None,
        is_private: bool = False,
        partner: Optional[tuple[str, str]] = None,
        tz_lookup: Optional[Callable[[int], Optional[str]]] = None,
        partner_user_id: Optional[int] = None,
    ) -> str:
        m = model or self._model
        if m not in self.KNOWN_MODELS:
            raise ValueError(f"unknown model: {m!r}. known: {sorted(self.KNOWN_MODELS)}")

        mode = self.mode_for(m)

        display_tz: tzinfo = UTC
        partner_tz_name = "UTC"
        if mode == "chat" and is_private and partner_user_id is not None and tz_lookup is not None:
            tz_name = tz_lookup(partner_user_id)
            if tz_name:
                try:
                    display_tz = ZoneInfo(tz_name)
                    partner_tz_name = tz_name
                except Exception:
                    logger.warning("Bad tz %r for partner %s; falling back to UTC", tz_name, partner_user_id)

        template_key = "chat_private" if (mode == "chat" and is_private) else mode
        template = Template(self.SYSTEM_PROMPTS.get(template_key, ""))
        partner_username, partner_display_name = partner if partner else ("", "")

        if mode == "chat" and not is_private:
            user_tz_directory, dir_users = self._build_tz_directory(messages, tz_lookup)
        else:
            user_tz_directory, dir_users = "", 0

        system = template.safe_substitute(
            display_name=display_name,
            username=username,
            model_name=m,
            partner_username=partner_username,
            partner_display_name=partner_display_name,
            partner_tz=partner_tz_name,
            user_tz_directory=user_tz_directory,
        )

        rendered = render_history(messages, username, mode, display_tz=display_tz)
        logger.debug(
            "Completion request: model=%s mode=%s template=%s message_turns=%d window_tokens=%d display_tz=%s dir_users=%d",
            m, mode, template_key, len(rendered), window_tokens,
            getattr(display_tz, "key", str(display_tz)),
            dir_users,
        )
        response = self.client.messages.create(
            model=m,
            system=system,
            max_tokens=max_tokens or self._max_tokens,
            messages=rendered,
            cache_control={"type": "ephemeral"},
        )

        text_parts = [b.text for b in response.content if isinstance(b, TextBlock)]
        text = "".join(text_parts)
        block_types = [getattr(b, "type", "?") for b in response.content]
        usage = response.usage
        cache_write = usage.cache_creation_input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        actual_input = usage.input_tokens + cache_write + cache_read
        # Heuristic check: window_tokens is len//4+5 per message; compare to the
        # API's actual count so we can see how far off the estimate runs.
        # Ratio >1 means the heuristic undercounts (real input larger than we
        # think) — relevant for both TOKEN_BUDGET tuning and eviction sizing.
        ratio = actual_input / window_tokens if window_tokens else float("nan")
        logger.info(
            "Token usage: estimated=%d actual=%d (ratio=%.2f) [uncached=%d cache_write=%d cache_read=%d output=%d]",
            window_tokens, actual_input, ratio,
            usage.input_tokens, cache_write, cache_read, usage.output_tokens,
        )
        logger.debug(
            "Completion response: %d chars from %d text block(s) of %d total (types=%s)",
            len(text), len(text_parts), len(response.content), block_types,
        )
        return text
