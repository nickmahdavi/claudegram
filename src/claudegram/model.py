from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional
from string import Template

import anthropic
from anthropic import AsyncClient
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    ModelParam,
    TextBlock,
    TextBlockParam,
)

from .identity import UserInfo

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    text: str
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    block_count: int
    block_types: list[str]
    true_input: int = field(init=False)

    def __post_init__(self):
        self.true_input = self.input_tokens + self.cache_read + self.cache_write

class Model(StrEnum):
    OPUS_4_7 = "claude-opus-4-7"
    OPUS_4_6 = "claude-opus-4-6"
    OPUS_4_5 = "claude-opus-4-5"
    OPUS_4_1 = "claude-opus-4-1"
    OPUS_4_0 = "claude-opus-4-0"
    SONNET_4_6 = "claude-sonnet-4-6"
    SONNET_4_5 = "claude-sonnet-4-5"
    SONNET_4_0 = "claude-sonnet-4-0"
    HAIKU_4_5 = "claude-haiku-4-5"

class PromptMode(StrEnum):
    PREFILL = "prefill"
    CHAT = "chat"
    CHAT_PRIVATE = "chat_private"

NO_PREFILL = (Model.OPUS_4_7, Model.OPUS_4_6, Model.SONNET_4_6, Model.HAIKU_4_5)

SUPPORTED_MODELS = frozenset(Model)

MODEL_ALIASES: dict[str, Model] = {
    "op4.7": Model.OPUS_4_7,
    "op4.6": Model.OPUS_4_6,
    "op4.5": Model.OPUS_4_5,
    "op4.0": Model.OPUS_4_0,
    "op4": Model.OPUS_4_0,
    "s4.6": Model.SONNET_4_6,
    "s4.5": Model.SONNET_4_5,
    "s4.0": Model.SONNET_4_0,
    "s4": Model.SONNET_4_0,
    "h4.5": Model.HAIKU_4_5,
}

SYSTEM_PROMPTS = {
    PromptMode.PREFILL: "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with outputs of the commands.",
    PromptMode.CHAT: "You're an LLM in a group conversation. Messages from other participants are prefixed with their name + handle + UTC time + offset suffix (e.g. '14:32 +00'). You should just send your messages like normal (no prefix).\nSome messages carry extra context on the line(s) above the body: 're. <name (@handle) ...> text' means the sender is replying to that earlier message; '> <name ...> \"text\"' means they quoted a specific span of it; 'fwd. <name (@handle) ...>' means the message was forwarded and the tag is the *original* author, not the participant who reposted it. Forwards from hidden users or channels may omit the @handle or the timestamp.\nYour display name is $display_name, and your username is $username. Users might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name.\n$user_tz_directory\nHave fun!",
    PromptMode.CHAT_PRIVATE: "You're an LLM in a private (1:1) conversation with $partner_display_name (@$partner_username). Their messages appear in human / user turns prefixed with their name + handle + local time + offset suffix (e.g. '14:32 -04'). Timestamps are rendered in their timezone ($partner_tz). You should just send your messages like normal (no prefix).\nSome messages carry extra context on the line(s) above the body: 're. <name (@handle) ...> text' means they're replying to that earlier message; '> <name ...> \"text\"' means they quoted a specific span of it; 'fwd. <name (@handle) ...>' means the message was forwarded and the tag is the *original* author, not the person who sent it to you. Forwards from hidden users or channels may omit the @handle or the timestamp.\nYour display name is $display_name, and your username is $username. They might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name. Have fun!",
}

def get_prompt(
    prompt_template: Template,
    model: Model | str,
    bot_info: UserInfo,
    partner: Optional[UserInfo] = None,
    tz_directory: Optional[str] = None  #, **kwargs for user-defined prompt vars
    ) -> str:

    # some redundancy here
    # also will need to be reworked when we decide to properly handle unknown users
    partner_display_name = partner.display_name if partner else "Unknown"
    partner_username = partner.username if partner else "unknown"
    if partner and not partner.tz:
        logger.debug("No partner tz for private chat; using UTC")
    partner_tz = partner.tz.key if partner and partner.tz else 'UTC'
    user_tz_directory = tz_directory or ""

    return prompt_template.safe_substitute(
        display_name=bot_info.display_name,
        username=bot_info.username,
        model_name=model,
        partner_display_name=partner_display_name,
        partner_username=partner_username,
        partner_tz=partner_tz,
        user_tz_directory=user_tz_directory,
    )

_EPHEMERAL: CacheControlEphemeralParam = {"type": "ephemeral", "ttl": "1h"}


def _mark_last_message_for_cache(messages: list[MessageParam]) -> list[MessageParam]:
    """Return a shallow copy of `messages` with cache_control marked on the
    last block of the last message. Required because Anthropic's prompt cache
    is content-prefix-addressed: a hit requires the prefix up to a marker to
    be byte-identical to a prior request. Marking the tail caches the entire
    history+system through the user's latest turn, which subsequent calls
    will hit as long as Window's eviction policy hasn't shifted the prefix
    (see Window.EVICT_TARGET)."""
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        blocks: list[TextBlockParam] = [
            {"type": "text", "text": content, "cache_control": _EPHEMERAL}
        ]
        last["content"] = blocks
    elif isinstance(content, list) and content:
        new_blocks = list(content)
        tail = dict(new_blocks[-1])
        tail["cache_control"] = _EPHEMERAL
        new_blocks[-1] = tail  # type: ignore[assignment]
        last["content"] = new_blocks
    out[-1] = last  # type: ignore[assignment]
    return out


TRANSIENT_ERRORS = (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError)

async def complete(
    client: AsyncClient,
    model: ModelParam,
    system: str,
    messages: list[MessageParam],
    max_tokens: int,
    mcp_servers: Optional[list[dict]] = None,
    max_retries: int = 3,
    retry_delay: float = 3.0,
) -> ClaudeResponse:
    # Two explicit cache breakpoints, used together:
    #   1. End of the system prompt: small always-on cache. Survives Window
    #      evictions (system content is independent of the working set as long
    #      as build_tz_directory is fed Window.known_users() rather than the
    #      snapshot). For DMs the system prompt may be under the per-model
    #      minimum-cacheable-size, in which case this marker is silently
    #      dropped -- harmless.
    #   2. End of the last message: caches the full conversation prefix.
    #      Hits during a burst of pings within the 1-hour TTL, until Window
    #      evicts and the prefix shifts.
    # Replaces the top-level `cache_control={"type":"ephemeral"}` kwarg, which
    # placed only the second marker and lost the system-prompt cache.
    system_blocks: list[TextBlockParam] = [
        {"type": "text", "text": system, "cache_control": _EPHEMERAL}
    ]
    cached_messages = _mark_last_message_for_cache(messages)

    for attempt in range(max_retries + 1):
        try:
            if mcp_servers:
                response = await client.beta.messages.create(
                    model=model,
                    system=system_blocks,
                    messages=cached_messages,
                    max_tokens=max_tokens,
                    mcp_servers=mcp_servers,  # type: ignore[arg-type]
                    betas=["mcp-client-2025-04-04"],
                )
            else:
                response = await client.messages.create(
                    model=model,
                    system=system_blocks,
                    messages=cached_messages,
                    max_tokens=max_tokens,
                )
            break
        except TRANSIENT_ERRORS as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "Completion attempt %d/%d failed (%s: %s); retrying in %.0fs",
                attempt + 1, max_retries + 1, type(exc).__name__, exc, retry_delay,
            )
            await asyncio.sleep(retry_delay)

    text_parts = [b.text for b in response.content if hasattr(b, "text") and b.type == "text"]
    text = "".join(text_parts)
    block_types = [getattr(b, "type", "?") for b in response.content]
    usage = response.usage

    logger.debug(
        "Completion response: %d chars from %d text block(s) of %d total (types=%s)",
        len(text), len(text_parts), len(response.content), block_types
    )
    
    return ClaudeResponse(
        text=text,
        input_tokens=usage.input_tokens,
        cache_write=usage.cache_creation_input_tokens or 0,
        cache_read=usage.cache_read_input_tokens or 0,
        output_tokens=usage.output_tokens,
        block_count=len(text_parts),
        block_types=block_types
    )
