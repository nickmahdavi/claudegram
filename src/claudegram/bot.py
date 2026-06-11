import asyncio
import functools
import json
import logging
import os
import re
import time
from pathlib import Path
from string import Template
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Coroutine, Literal, Optional
from zoneinfo import ZoneInfo, available_timezones

import telegram
from telegram import Update, User
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .commands import ActiveChat, CaptionCommand, MentionsMe, NotIgnored, RepliesToMe
from .config import Config
from .credentials import (
    USER_OWNED_KINDS,
    BillingMode,
    Credential,
    CredentialKind,
    CredentialStore,
)
from .error import (
    ErrorClass,
    admin_failure_dm,
    admin_recovery_dm,
    classify_error,
    credential_broken_reply,
    designated_credential_failed_reply,
    no_credential_reply,
    user_credential_failed_reply,
    user_reply,
)
from .identity import UserInfo
from .importer import parse_export
from .message import UTC, Forward, Message, Reply
from .mcp import McpTokenManager
from .model import MODEL_ALIASES, TRANSIENT_ERRORS, SUPPORTED_MODELS, SYSTEM_PROMPTS, PromptMode, complete, get_prompt
from .render import RenderMode, build_tz_directory, fmt_offset, render_history
from .transport import CommandCtx, Incoming, Outgoing
from .store import Store

# Bot API caps document downloads at 20 MB
LOAD_MAX_BYTES = 18 * 1024 * 1024
# Discard pings older than this many seconds
STALE_PING_AGE_S = 60
TELEGRAM_CHAR_LIMIT = 4096
# Coalesce per-chat view-log rewrites to at most one per this interval. A busy
# group otherwise re-renders the whole window + writes a file on every message;
# the bot only actually "sees" history at ping time, which forces a write anyway.
VIEW_LOG_MIN_INTERVAL_S = 1.0
TIMEZONES: frozenset[str] = frozenset(available_timezones())

logger = logging.getLogger(__name__)

# Anthropic API keys / OAuth tokens look like `sk-ant-...` followed by a long
# token body. Require the trailing body (>=10 url-safe chars) so we don't nuke a
# DM that merely *mentions* the "sk-ant-" prefix while still catching a pasted
# key anywhere in the message (incl. mid-sentence or in a caption).
_SECRET_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")


def _looks_like_secret(text: str) -> bool:
    return _SECRET_RE.search(text) is not None


def get_user_info(user: User) -> UserInfo:
    return UserInfo(
        user_id=user.id,
        username=user.username or "",
        display_name=user.full_name,
    )

def get_forward(message: telegram.Message) -> Optional[Forward]:
    origin = message.forward_origin
    if origin is None:
        return None
    if isinstance(origin, telegram.MessageOriginUser):
        u = origin.sender_user
        return Forward(
            display_name=u.full_name or u.username or f"user_{u.id}",
            username=u.username or "",
            ts=origin.date,
            user_id=u.id,
        )
    if isinstance(origin, telegram.MessageOriginHiddenUser):
        return Forward(
            display_name=origin.sender_user_name,
            username="",
            ts=origin.date,
            user_id=None,
        )
    if isinstance(origin, telegram.MessageOriginChat):
        chat = origin.sender_chat
        # sender_chat is a group/supergroup/channel acting as sender, so it has a
        # title, not full_name (the latter is a deprecated user-only alias).
        name = chat.title or chat.username or "chat"
        if origin.author_signature:
            name = f"{name} ({origin.author_signature})"
        return Forward(
            display_name=name,
            username=chat.username or "",
            ts=origin.date,
            user_id=None,
        )
    if isinstance(origin, telegram.MessageOriginChannel):
        chat = origin.chat
        name = chat.title or chat.username or "channel"
        if origin.author_signature:
            name = f"{name} ({origin.author_signature})"
        return Forward(
            display_name=name,
            username=chat.username or "",
            ts=origin.date,
            user_id=None,
        )
    logger.warning("Unknown forward_origin type %r in chat %s", type(origin).__name__, message.chat_id)
    return None

@asynccontextmanager
async def keep_typing(bot, chat_id: int, interval: float = 4.0):
    async def loop():
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=telegram.constants.ChatAction.TYPING)
            except Exception:
                logger.debug("Typing keep-alive stopped (chat %s)", chat_id)
                return
            await asyncio.sleep(interval)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

def incoming(func: Callable[["Bot", Incoming], Awaitable[Optional[Outgoing]]]) -> Callable[["Bot", Update, ContextTypes.DEFAULT_TYPE], Coroutine[object, object, None]]:
    """Adapt a pure Incoming -> Optional[Outgoing] handler into a PTB handler.

    Pure ingress: project the Update into an Incoming and (if the handler
    returns one) send the Outgoing. All gating — active chat, mention/reply,
    chat type — is done declaratively by the registered filters, so anything
    reaching here has already qualified; we only drop the structurally
    unusable (no message/sender/text) and the ignore-prefix escape hatch.

    A returned Outgoing is a simple framework reply (refusal, error) — sent
    with the system prefix, not persisted. Handlers that need to persist model
    output to history call self._send(...) themselves and return None."""
    @functools.wraps(func)
    async def handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message or not message.from_user:
            return

        text = message.text or message.caption
        if not text:
            return

        # Belt-and-suspenders: on_ping's registration already gates on the
        # NotIgnored filter, but keeping the check here means @incoming stays
        # correct for any future handler registered without that filter.
        if self.config.ignore_prefix and text.startswith(self.config.ignore_prefix):
            return

        incoming = Incoming(
            sender=get_user_info(message.from_user),
            message_id=message.message_id,
            date=message.date,
            chat_id=message.chat_id,
            text=text,
            is_private=(message.chat.type == telegram.constants.ChatType.PRIVATE),
        )

        outgoing = await func(self, incoming)

        if outgoing is None:
            return

        prefix = f"{self.config.system_prefix} " if outgoing.system else ""
        await message.reply_text(f"{prefix}{outgoing.text}")
    return handler


def command(
    *, admin: Literal["never", "in_groups", "always"] = "never",
) -> Callable[
    [Callable[["Bot", CommandCtx], Awaitable[None]]],
    Callable[["Bot", Update, ContextTypes.DEFAULT_TYPE], Coroutine[object, object, None]],
]:
    """Adapt a (self, CommandCtx) -> None handler into a PTB command callback.

    Absorbs the message/user extraction every command repeats, projects a
    CommandCtx, and applies the admin gate:
      - "never":     no gate (default)
      - "in_groups": DMs ok, groups require admin (start/stop/load/model/billing)
      - "always":    admin everywhere, even DMs (allow/disallow/poollist)
    A denied invocation LOGS (the truthful once-per-rejection audit a filter
    can't give — that's why the gate lives here, not in a filter) and returns
    silently to the user, matching today's no-leak behavior."""
    def decorate(func):
        @functools.wraps(func)
        async def handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            message = update.message
            user = update.effective_user
            if not message or not user:
                return

            is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
            is_admin = user.id in self.config.admin_user_ids

            denied = (
                (admin == "always" and not is_admin)
                or (admin == "in_groups" and not is_private and not is_admin)
            )
            if denied:
                logger.warning(
                    "Non-admin /%s denied in chat %s (user_id=%s)",
                    func.__name__.removeprefix("command_"), message.chat_id, user.id,
                )
                return

            ctx = CommandCtx(
                message=message,
                user=user,
                chat_id=message.chat_id,
                is_private=is_private,
                args=context.args or [],
                is_admin=is_admin,
                update=update,
                context=context,
            )
            await func(self, ctx)
        return handler
    return decorate


class Bot:
    def __init__(self, store: Store, config: Config, credentials: CredentialStore):
        self.store = store
        self.config = config
        self.credentials = credentials

        self._me: Optional[UserInfo] = None

        # The shared bot/pool key lives in the credential store; keep a handle
        # for any direct use (e.g. legacy call sites). Per-request clients are
        # resolved via self.credentials.client_for(...).
        self.client = credentials.pool_client

        self.mcp_tokens: Optional[McpTokenManager] = (
            McpTokenManager(
                token_url=config.mcp_token_url,
                client_id=config.mcp_client_id,
                client_secret=config.mcp_client_secret,
            )
            if config.mcp_enabled
            else None
        )
        self.application = (
            Application.builder()
            .token(self.config.telegram_bot_token)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )

        self.start_time = datetime.now(UTC)

        self._prompt_changelog_path = Path(config.data_dir) / "prompt_changelog.jsonl"
        self._prompt_changelog: list[dict] = self._load_prompt_changelog()

        # View-log bookkeeping: per-chat throttle clock + warn-once dedupe so a
        # persistently failing view write (e.g. unwritable log_dir) is surfaced
        # once instead of on every message.
        self._view_last_write: dict[int, float] = {}
        self._view_warned: set[int] = set()

    async def _post_init(self, _: Application):
        try:
            me = await self.application.bot.get_me()
            self._me = get_user_info(me)
            self.store.note_user(self._me)
        except Exception as e:
            logger.error("Failed to get bot ID: %s", e, exc_info=True)
            raise e
        logger.info("Bot initialized: id=%d default_model=%r ignore_prefix=%r admins=%d start_time=%s",
                    self.me.user_id, self.config.default_claude_model, self.config.ignore_prefix,
                    len(self.config.admin_user_ids), self.start_time.isoformat())

    async def _post_shutdown(self, _: Application):
        # Close every per-user httpx pool (and the shared pool client) exactly
        # once, at shutdown — never mid-run, where an evicted client might still
        # be serving an in-flight completion on another coroutine.
        await self.credentials.aclose()

    @property
    def me(self) -> UserInfo:
        if self._me is None:
            raise RuntimeError("Bot identity not loaded yet (app not initialized?)")
        return self._me

    async def _send(
        self,
        reply: Callable[[str], Awaitable[Optional[telegram.Message]]],
        incoming: Incoming,
        text: str,
        incarnation: int,
        display_tz: Optional[ZoneInfo],
    ):
        """Send a model reply (chunked to Telegram's limit) and, if the chat's
        history hasn't been swapped out from under us since we snapshotted
        (incarnation match), persist it. `reply` is the bound send callable
        (e.g. message.reply_text) so this stays decoupled from the Update."""
        chunks = [text[i:i + TELEGRAM_CHAR_LIMIT] for i in range(0, len(text), TELEGRAM_CHAR_LIMIT)]
        if len(chunks) > 1:
            logger.info(
                "Chunking reply for chat %s (%d chars -> %d pieces)",
                incoming.chat_id, len(text), len(chunks),
            )
        async with self.store.lock(incoming.chat_id):
            # A concurrent /reset or /load while we were awaiting the model
            # would have wiped or replaced this chat's history. If its
            # incarnation moved since we snapshotted, the reply we computed is
            # for a conversation that no longer exists: still send it (the user
            # pinged and deserves an answer) but don't persist it onto the new
            # window. We hold the lock across the whole send loop, so
            # the incarnation can't change again mid-loop.
            # 
            # This is currently bullshit in 99% of cases
            stale = self.store.incarnation(incoming.chat_id) != incarnation
            if stale:
                logger.warning(
                    "Chat %s history changed during completion (incarnation %d -> %d); "
                    "sending reply without persisting",
                    incoming.chat_id, incarnation, self.store.incarnation(incoming.chat_id),
                )
            window = None if stale else self.store.window(incoming.chat_id)
            for i, chunk in enumerate(chunks):
                first = i == 0
                sent = await reply(chunk)
                if sent is None or sent.text is None:
                    logger.warning("Chunk %d not sent for chat %s; skipping persist", i, incoming.chat_id)
                elif window is not None:
                    window.append(Message(
                        id=sent.message_id,
                        ts=sent.date,
                        user_id=self.me.user_id,
                        text=sent.text,
                        reply_to=incoming.message_id if first else None,
                        reply= Reply(
                            user_id=incoming.sender.user_id,
                            text=incoming.text,
                            is_quote=False,
                            ts=incoming.date
                        ) if first else None
                    ))
            view_snapshot = None
            if window is not None:
                self.store.persist(incoming.chat_id)
                if self._should_write_view(incoming.chat_id, force=True):
                    view_snapshot = window.snapshot()

        if view_snapshot is not None:
            await self._write_chat_view(incoming.chat_id, view_snapshot, display_tz, incoming.is_private, incoming.sender.user_id)

    async def _say(self, ctx: CommandCtx, text: str, *, markdown: bool = True) -> None:
        """Reply to a command with the system prefix. markdown=True is the
        common case; the credential/pool/billing commands pass markdown=False
        because they interpolate user-controlled display names, and Markdown
        would let a name like *foo* break the whole send."""
        await ctx.message.reply_text(
            f"{self.config.system_prefix} {text}",
            parse_mode="Markdown" if markdown else None,
        )

    def _should_write_view(self, chat_id: int, force: bool) -> bool:
        """Throttle gate for the view log. Returns True (and arms the clock) when a
        write should happen now. `force=True` (the ping render -- the moment the bot
        actually sees the history) always writes. Pure/synchronous: call it under the
        chat lock before taking a snapshot so we only snapshot when we'll write."""
        if not self.config.chat_view_log:
            return False
        now = time.monotonic()
        if not force and now - self._view_last_write.get(chat_id, 0.0) < VIEW_LOG_MIN_INTERVAL_S:
            return False
        self._view_last_write[chat_id] = now
        return True

    def _load_prompt_changelog(self) -> list[dict]:
        try:
            lines = self._prompt_changelog_path.read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]
        except FileNotFoundError:
            return []
        except Exception:
            logger.exception("Failed to load prompt changelog from %s", self._prompt_changelog_path)
            return []

    def _save_prompt_changelog(self) -> None:
        self._prompt_changelog_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(entry) for entry in self._prompt_changelog)
        self._prompt_changelog_path.write_text(text + "\n" if text else "", encoding="utf-8")

    def _render_changelog(self) -> str:
        return "\n\n".join(f"[{e['timestamp']}] {e['text']}" for e in self._prompt_changelog)

    def _build_system(self, base: str) -> str:
        rendered = self._render_changelog()
        changelog = rendered if rendered else "(no entries yet)"
        return f"{base}\n\n— Changelog —\n{changelog}"

    def _base_prompt(self, chat_id: int, snapshot: list[Message], is_private: bool, partner_id: Optional[int]) -> str:
        """Base system prompt for this chat, without the changelog."""
        prompt_mode = PromptMode.CHAT_PRIVATE if is_private else PromptMode.CHAT
        prompt_template = Template(SYSTEM_PROMPTS.get(prompt_mode, ""))
        chat_model = self.store.get_model_pref(chat_id) or self.config.default_claude_model
        partner = self.store.resolve_user(partner_id) if (is_private and partner_id is not None) else None
        tz_directory: Optional[str] = None
        if not is_private:
            known = {m.user_id for m in snapshot}
            known.discard(self.me.user_id)
            tz_directory = build_tz_directory(known, self.store.resolve_user)
        return get_prompt(
            prompt_template=prompt_template,
            model=chat_model,
            bot_info=self.me,
            partner=partner,
            tz_directory=tz_directory,
        )

    def _view_system_prompt(self, chat_id: int, snapshot: list[Message], is_private: bool, partner_id: Optional[int]) -> str:
        """Full system prompt as the model sees it (base + changelog)."""
        return self._build_system(self._base_prompt(chat_id, snapshot, is_private, partner_id))

    async def _write_chat_view(
        self, chat_id: int, snapshot: list[Message], display_tz: Optional[ZoneInfo],
        is_private: bool, partner_id: Optional[int],
    ) -> None:
        """Rewrite the per-chat 'as the bot sees it' log: a 'System:' header with the
        chat's current system prompt, followed by the window rendered into H:/A:
        turns -- the same shape sent to the model at ping time. Rendering runs on the
        loop (it reads the store's identity table, which is only safe on the loop);
        the file write is offloaded with asyncio.to_thread and done via a tmp +
        os.replace swap so a reader never sees a torn/empty file. Because the swap
        changes the inode, follow it with `tail -F` (not `-f`). Best-effort: a
        logging failure must never break message handling, and a persistent failure
        is warned about once per chat rather than on every write."""
        try:
            system = self._view_system_prompt(chat_id, snapshot, is_private, partner_id)
            messages = render_history(snapshot, self.me, RenderMode.CHAT, self.store.resolve_user, display_tz)
            blocks = [f"System: {system}"]
            for mp in messages:
                # MessageParam is a TypedDict (plain dict at runtime), not an attr object.
                prefix = "H:" if mp["role"] == "user" else "A:"
                content = mp["content"]
                content = content if isinstance(content, str) else str(content)
                blocks.append(f"{prefix} {content}")
            text = "\n\n".join(blocks) + "\n"
            await asyncio.to_thread(self._write_view_file, self.store.view_path(chat_id), text)
            self._view_warned.discard(chat_id)
        except Exception:
            if chat_id not in self._view_warned:
                self._view_warned.add(chat_id)
                logger.warning("Failed to write chat view log for chat %s (further failures suppressed)",
                               chat_id, exc_info=True)

    @staticmethod
    def _write_view_file(path, text: str) -> None:
        """Atomic overwrite (runs in a worker thread): write to a sibling tmp file
        then os.replace it into place."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    async def _drop_pasted_secret(self, message: telegram.Message) -> None:
        """Delete a message that contains an API key, refuse, and warn. Shared by
        the new-message and edited-message guards. Callers raise
        ApplicationHandlerStop afterwards so no later handler group sees it."""
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        logger.warning(
            "Suspected API key in %s chat %s from user %s; dropping unstored",
            message.chat.type, message.chat_id,
            message.from_user.id if message.from_user else None,
        )
        deleted = False
        try:
            deleted = bool(await message.delete())
        except Exception:
            logger.warning("Could not delete suspected-key message in chat %s", message.chat_id)
        prefix = self.config.system_prefix
        if is_private:
            note = "I deleted it and did not store it." if deleted else "I did not store it (couldn't delete it — please remove it)."
            await message.reply_text(
                f"{prefix} That looked like an API key, so {note} Use /setkey <key> to register it securely."
            )
        else:
            note = "I deleted it." if deleted else "Please delete it — I couldn't."
            await message.reply_text(
                f"{prefix} Don't paste API keys in a group. {note} DM me /setkey <key> in private instead."
            )

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        # why is from_user nullable at all?
        if not message or not message.from_user:
            return

        text = message.text or message.caption
        if not text:
            return

        # Secret guard — FIRST, in EVERY chat type, and BEFORE the is_active /
        # ignore_prefix gates. A pasted key (incl. as a media caption, which
        # filters.COMMAND doesn't exclude so /setkey-as-caption lands here too)
        # would otherwise be written to chat history in plaintext AND, if it
        # pings the bot, rendered into the prompt and echoed in the reply. Group
        # chats are just as exposed as DMs. Delete it, refuse, and raise
        # ApplicationHandlerStop so on_ping (a later handler group) never sees it
        # (a bare `return` only ends THIS handler).
        if _looks_like_secret(text):
            await self._drop_pasted_secret(message)
            raise ApplicationHandlerStop

        if not self.store.is_active(message.chat_id):
            return

        if self.config.ignore_prefix and text.startswith(self.config.ignore_prefix):
            logger.debug("Skipping ignored message (prefix=%r) in chat %s", self.config.ignore_prefix, message.chat_id)
            return

        self.store.note_user(get_user_info(message.from_user))

        reply = None
        replied = message.reply_to_message
        # why is from_user nullable at all?
        if replied and replied.from_user:
            self.store.note_user(get_user_info(replied.from_user))
            reply = Reply(
                user_id=replied.from_user.id,
                text=message.quote.text if message.quote else replied.text or replied.caption or "",
                is_quote=message.quote is not None,
                ts=replied.date,
            )

        try:
            forward = get_forward(message)
        except Exception:
            logger.warning("Failed to extract forward origin in chat %s; storing without it",
                           message.chat_id, exc_info=True)
            forward = None

        msg = Message(
            id=message.message_id,
            ts=message.date,
            user_id=message.from_user.id,
            text=text,
            reply_to=replied.message_id if replied else None,
            reply=reply,
            forward=forward,
        )

        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            evicted = window.append(msg)
            if evicted:
                logger.debug("Evicted %d message(s) from working set in chat %s", len(evicted), message.chat_id)
            self.store.persist(message.chat_id)
            # Only snapshot when we'll actually write -- snapshot() copies the window.
            view_snapshot = window.snapshot() if self._should_write_view(message.chat_id, force=False) else None

        if view_snapshot is not None:
            is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
            view_tz = self.store.resolve_user(message.from_user.id).tz if is_private else UTC
            await self._write_chat_view(message.chat_id, view_snapshot, view_tz, is_private, message.from_user.id)

    async def on_edited_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """A user editing a benign message to insert a key arrives as an
        edited_message, which on_message/on_ping (both read update.message) never
        process — so the key wouldn't be stored, but also wouldn't be cleaned up.
        Apply the same secret guard here. Registered ahead of on_message in the
        same group with an EDITED_MESSAGE-only filter, so normal messages still
        fall through to on_message."""
        message = update.edited_message
        if not message:
            return
        text = message.text or message.caption
        if text and _looks_like_secret(text):
            await self._drop_pasted_secret(message)
            raise ApplicationHandlerStop

    @incoming
    async def on_ping(self, incoming: Incoming) -> Optional[Outgoing]:
        # Reply by chat_id rather than against a Message object: the @incoming
        # decorator hands us a transport-agnostic Incoming, and send_message
        # threaded with reply_to keeps the same threading reply_text gave us.
        bot = self.application.bot

        async def reply(text: str) -> Optional[telegram.Message]:
            return await bot.send_message(
                chat_id=incoming.chat_id,
                text=text,
                reply_to_message_id=incoming.message_id,
            )

        prompt_mode = PromptMode.CHAT_PRIVATE if incoming.is_private else PromptMode.CHAT
        render_mode = RenderMode.CHAT  # TODO Prefill is borked atm, will destub later

        logger.info("on_ping triggered in chat %s", incoming.chat_id)

        # telegram buffers updates for 24h and for now I think the best design is to avoid spamming
        # users after the bot comes back up following an outage. but in the future we should buffer
        # these messages locally and allow the user to decide whether to process them or not,
        # probably in bulk with a single command. and this should also be unified with missed
        # buffered messages while out of credits, during anthropic API outages, etc. for now, since
        # we aren't handling or resending these, it's consistent to just drop stale messages
        
        now = datetime.now(UTC)
        last_load = self.store.get_last_load_at(incoming.chat_id)
        cutoff = max(filter(None, [self.start_time, now - timedelta(seconds=STALE_PING_AGE_S), last_load]))
        if incoming.date < cutoff:
            age = (now - incoming.date).total_seconds()
            logger.info(
                "Skipping stale ping in chat %s (age=%.1fs, cutoff=%s)",
                incoming.chat_id, age, cutoff.isoformat(),
            )
            return

        # Resolve whose credential pays for this ping. None => polite refusal,
        # and crucially we must NOT touch the failure-streak/admin-alert
        # machinery (that tracks only the bot's pool key). Done before any
        # prompt building or typing indicator so a refusal is cheap and silent.
        cred = self.credentials.resolve_credential(incoming.sender.user_id, incoming.chat_id, incoming.is_private)
        if cred is None:
            logger.info("No credential for user %s in chat %s; refusing", incoming.sender.user_id, incoming.chat_id)
            return Outgoing(
                text=no_credential_reply(incoming.is_private),
                system=True
            )
        try:
            client = self.credentials.client_for(cred)
        except Exception:
            logger.exception(
                "Failed to build client for cred kind=%s user=%s", cred.kind.value, incoming.sender.user_id,
            )
            return Outgoing(
                text=credential_broken_reply(),
                system=True
            )

        prompt_template = Template(SYSTEM_PROMPTS.get(prompt_mode, ""))
        chat_model = self.store.get_model_pref(incoming.chat_id) or self.config.default_claude_model
        partner: Optional[UserInfo] = None
        display_tz: Optional[ZoneInfo] = UTC
        tz_directory: Optional[str] = None

        async with self.store.lock(incoming.chat_id):
            window = self.store.window(incoming.chat_id)
            snapshot = window.snapshot()
            window_tokens = window.tokens
            known_users = window.known_users()
            incarnation = self.store.incarnation(incoming.chat_id)

        if prompt_mode == PromptMode.CHAT_PRIVATE:
            # might be weird defaulting to anonymous in dms?
            # maybe we should add an unknown user distinct from anon
            partner = self.store.resolve_user(incoming.sender.user_id)
            display_tz = partner.tz
        else:
            known_users.discard(self.me.user_id)  # don't show me in my own directory
            tz_directory = build_tz_directory(known_users, self.store.resolve_user)

        system = self._build_system(get_prompt(
            prompt_template=prompt_template,
            model=chat_model,
            bot_info=self.me,
            partner=partner,
            tz_directory=tz_directory
        ))

        messages = render_history(snapshot, self.me, render_mode, self.store.resolve_user, display_tz)

        mcp_servers = None
        if self.mcp_tokens is not None:
            token = await self.mcp_tokens.get_token()
            if token is not None:
                mcp_servers = [{
                    "type": "url",
                    "url": self.config.mcp_server_url,
                    "name": self.config.mcp_server_name,
                    "authorization_token": token,
                }]

        async with keep_typing(bot, incoming.chat_id):
            try:
                completion = await complete(
                    client=client,
                    model=chat_model,
                    system=system,
                    messages=messages,
                    max_tokens=self.config.reply_budget,
                    mcp_servers=mcp_servers,
                )
            except TRANSIENT_ERRORS as exc:
                if mcp_servers is None:
                    await self._handle_completion_error(incoming, reply, exc, cred)
                    return
                # MCP server unreachable after all retries — fall back to a
                # completion without tools, but tell Claude what happened so he
                # can acknowledge it rather than silently losing persistence.
                logger.warning("MCP unavailable after retries; falling back to non-MCP completion: %s", exc)
                await reply(f"{self.config.system_prefix} Memory tools temporarily unavailable — continuing without them.")
                try:
                    completion = await complete(
                        client=client,
                        model=chat_model,
                        system=system,
                        messages=messages,
                        max_tokens=self.config.reply_budget,
                        mcp_servers=None,
                    )
                except Exception as fallback_exc:
                    await self._handle_completion_error(incoming, reply, fallback_exc, cred)
                    return
            except Exception as exc:
                await self._handle_completion_error(incoming, reply, exc, cred)
                return

            # window_tokens is len // 4 + 5 + (overhead) per message. Empirically the ratio
            # here is about 1.05, but it's different for opus 4.7 and likely later
            # models, so good to keep it around
            ratio = completion.true_input / window_tokens if window_tokens else float("nan")
            logger.info(
                "Token usage for %s: estimated=%d actual=%d (ratio=%.2f) "\
                "[uncached=%d cache_write=%d cache_read=%d output=%d]",
                chat_model, window_tokens, completion.true_input, ratio,
                completion.input_tokens, completion.cache_write, completion.cache_read, completion.output_tokens,
            )

        # Only a POOL (bot-key) success clears the chat's failure streak.
        # Clearing it on a user-key success would let one BYO user mask an
        # ongoing pool outage and flap the recovery/outage admin alerts.
        prior_failures = 0
        if cred.kind == CredentialKind.POOL:
            async with self.store.lock(incoming.chat_id):
                prior_failures = self.store.note_success(incoming.chat_id)
        if prior_failures:
            await self._notify_admins_recovered(incoming.chat_id, prior_failures)

        if not completion.text:
            logger.warning("Model returned empty reply for chat %s", incoming.chat_id)
            return

        await self._send(reply, incoming, completion.text, incarnation, display_tz)
        return None


    async def _handle_completion_error(
        self,
        incoming: Incoming,
        reply: Callable[[str], Awaitable[Optional[telegram.Message]]],
        exc: Exception,
        cred: Credential,
    ) -> None:
        chat_id = incoming.chat_id
        err_class, err_desc = classify_error(exc)

        # Per-user failure isolation. A failure on a user's OWN credential
        # (bad/expired key, out-of-credit, their account rate-limited, etc.)
        # must be attributed to that user and must NOT feed the chat's
        # failure-streak / admin-alert machinery, which tracks only the bot's
        # pool key. Out-of-credit (402) is the most common BYO failure and the
        # easiest to misattribute as "the bot is down".
        if cred.kind in USER_OWNED_KINDS:
            # Attribute to the credential's OWNER (cred.user_id), not the user
            # who triggered the ping. In CHAT_DESIGNATED billing the payer is
            # the chat's designated user, who may be someone other than the
            # triggerer; logging incoming.sender here would point at the wrong
            # account.
            triggered_by_payer = incoming.sender.user_id == cred.user_id
            logger.warning(
                "User-owned credential failure (payer=%s, triggered_by=%s, kind=%s, class=%s, desc=%s)",
                cred.user_id, incoming.sender.user_id,
                cred.kind.value, err_class.value, err_desc,
            )
            if triggered_by_payer:
                text = user_credential_failed_reply(err_class)
            else:
                # Designated payer != triggerer: name the payer instead of
                # telling the triggerer "your key failed" (theirs is fine, or
                # they have none). The reply is public in the group, so this
                # also routes the actionable info to the person who can fix it.
                payer = self.store.resolve_user(cred.user_id).display_name
                text = designated_credential_failed_reply(err_class, payer)
            try:
                await reply(f"{self.config.system_prefix} {text}")
            except Exception:
                logger.exception("Failed to send per-user credential error reply in chat %s", chat_id)
            return

        # Serialize the whole failure-state transition for this chat. Without
        # the lock, two concurrent failing pings could both clear the
        # should_send/should_alert gates and double-send error replies or
        # double-DM admins (breaking "alert once per streak") -- these are
        # check-then-act sequences straddling awaits. The sends stay inside the
        # lock too: the error path is infrequent and it's the same chat that's
        # already failing, so serializing it is cheap. Not nested under any
        # other store.lock acquisition (the ping released its snapshot lock
        # before we got here), so no re-entrant deadlock on the non-reentrant
        # asyncio.Lock.
        async with self.store.lock(chat_id):
            count = self.store.note_failure(chat_id)
            logger.warning(
                "Completion failed in chat %s (class=%s, desc=%s, consecutive=%d)",
                chat_id, err_class.value, err_desc, count,
                exc_info=exc,
            )

            if self.store.should_send_error_reply(chat_id):
                text = f"{self.config.system_prefix} {user_reply(err_class)}"
                try:
                    await reply(text)
                    self.store.mark_error_reply_sent(chat_id)
                except Exception:
                    logger.exception("Failed to send error reply in chat %s", chat_id)

            if self.store.should_alert_admin(chat_id):
                await self._notify_admins_failure(chat_id, err_class, err_desc, count)
                self.store.mark_admin_alerted(chat_id)

    async def _notify_admins_failure(
        self, chat_id: int, err_class: ErrorClass, err_desc: str, count: int,
    ) -> None:
        if not self.config.admin_user_ids:
            logger.warning(
                "Failure streak in chat %s hit alert threshold, but no admins configured",
                chat_id,
            )
            return
        text = f"{self.config.system_prefix} {admin_failure_dm(err_class, chat_id, count, err_desc)}"
        for admin_id in self.config.admin_user_ids:
            try:
                # no md-- don't eat backticks
                await self.application.bot.send_message(chat_id=admin_id, text=text)
            except Exception:
                logger.exception("Failed to DM admin %s about failures in chat %s", admin_id, chat_id)

    async def _notify_admins_recovered(self, chat_id: int, prior_count: int) -> None:
        if not self.config.admin_user_ids:
            return
        text = f"{self.config.system_prefix} {admin_recovery_dm(chat_id, prior_count)}"
        for admin_id in self.config.admin_user_ids:
            try:
                # no md-- don't eat backticks
                await self.application.bot.send_message(chat_id=admin_id, text=text)
            except Exception:
                logger.exception("Failed to DM admin %s about recovery in chat %s", admin_id, chat_id)

    @command(admin="in_groups")
    async def command_start(self, ctx: CommandCtx):
        if self.store.set_active(ctx.chat_id, True):
            await self._say(ctx, "Already listening here. `/stop` to pause.")
        else:
            if ctx.is_private:
                hint = "I'll respond to every message."
            else:
                hint = f"Mention me with @{self.me.username} or reply to my messages for a response."
            await self._say(
                ctx,
                f"Hi! I'm {self.me.display_name}. {hint}\n"
                f"`/stop` to stop listening, `/help` for commands.",
            )
            logger.info("Activated chat %s by user_id=%s", ctx.chat_id, ctx.user.id)

    @command(admin="in_groups")
    async def command_stop(self, ctx: CommandCtx):
        if self.store.set_active(ctx.chat_id, False):
            await self._say(ctx, "Going silent. `/start` to re-enable.")
            logger.info("Deactivated chat %s by user_id=%s", ctx.chat_id, ctx.user.id)
        else:
            await self._say(ctx, "Already inactive here.")

    @command(admin="in_groups")
    async def command_reset(self, ctx: CommandCtx):
        async with self.store.lock(ctx.chat_id):
            window, file, context = self.store.reset(ctx.chat_id)

        if window or file or context:
            logger.info("Reset chat %s (window=%s, file=%s, ctx=%s)", ctx.chat_id, window, file, context)
            await self._say(ctx, "History cleared.")
        else:
            logger.debug("Reset called on empty chat %s", ctx.chat_id)
            await self._say(ctx, "Nothing to clear.")

    @command()
    async def command_whoami(self, ctx: CommandCtx):
        suffix = " (admin)" if ctx.is_admin else ""
        await self._say(ctx, f"Your telegram user id: `{ctx.user.id}`{suffix}")

    @command(admin="in_groups")
    async def command_save(self, ctx: CommandCtx):
        async with self.store.lock(ctx.chat_id):
            n = self.store.persist(ctx.chat_id)
        if n:
            await self._say(ctx, f"Saved {n} new message{"s" if n != 1 else ""}.")
        else:
            await self._say(ctx, "Nothing new to save.")

    @command(admin="in_groups")
    async def command_load(self, ctx: CommandCtx):
        chat_id = ctx.chat_id
        message = ctx.message

        document = message.document
        if document is None:
            await self._say(
                ctx,
                "Attach `result.json` from a Telegram Desktop chat export with caption `/load`.",
            )
            return

        filename = (document.file_name or "").lower()
        mime = document.mime_type or ""
        if not (filename.endswith(".json") or mime in ("application/json", "text/plain")):
            await self._say(
                ctx,
                f"Expected a `.json` file (got `{document.file_name}`, mime `{mime}`).",
            )
            return

        if document.file_size and document.file_size > LOAD_MAX_BYTES:
            await self._say(
                ctx,
                f"File too large ({document.file_size:,} bytes; limit is {LOAD_MAX_BYTES:,}).",
                markdown=False,
            )
            return

        async with self.store.lock(chat_id):
            tmp_dir = self.store.data_dir / "imports"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"chat_{chat_id}.{int(time.time())}.import.json"
            try:
                tg_file = await document.get_file()
                await tg_file.download_to_drive(tmp_path)
                logger.info(
                    "Downloaded /load payload for chat %s to %s (%s bytes)",
                    chat_id, tmp_path, document.file_size,
                )

                result = await asyncio.to_thread(
                    parse_export,
                    tmp_path,
                    self.me,
                    self.config.system_prefix
                )

                if result.kept == 0:
                    await self._say(
                        ctx,
                        f"Parsed {result.total} messages, kept 0 "
                        f"(dropped: {result.dropped_system} system, "
                        f"{result.dropped_commands} commands, "
                        f"{result.dropped_service} service, "
                        f"{result.dropped_non_text} non-text). "
                        f"Refusing to replace history with nothing.",
                        markdown=False,
                    )
                    return

                # Seed display names recovered from the export so imported
                # history renders with real names rather than the "User <id>"
                # fallback. Only for users we don't already know: live data
                # (real @handles, timezones) must win over the export's
                # name-only, placeholder-handle identities.
                seeded = 0
                for uid, uinfo in result.users.items():
                    if self.store.get_user(uid) is None:
                        self.store.note_user(uinfo)
                        seeded += 1
                if seeded:
                    logger.info("Seeded %d new user identit%s from import for chat %s",
                                seeded, "y" if seeded == 1 else "ies", chat_id)

                backup = self.store.replace_chat(chat_id, result.messages)
                self.store.mark_loaded(chat_id)
            except Exception as e:
                logger.exception("Load failed for chat %s", chat_id)
                await self._say(ctx, f"Load failed: {e}", markdown=False)
                return
            finally:
                tmp_path.unlink(missing_ok=True)

        await self._say(
            ctx,
            f"Loaded {result.kept}/{result.total} messages "
            f"(dropped: {result.dropped_system} system, "
            f"{result.dropped_commands} commands, "
            f"{result.dropped_service} service, "
            f"{result.dropped_non_text} non-text). "
            f"Backup: `{backup.name}`",
        )

    @command()
    async def command_tz(self, ctx: CommandCtx):
        user = ctx.user
        # In-body, not a decorator gate: this rejection REPLIES, so it isn't a
        # silent drop the decorator/filters can own.
        if user.is_bot:
            await self._say(ctx, "I can't set a timezone for a bot/anonymous user. Try starting a private chat with me.", markdown=False)
            return

        args = ctx.args
        user_info = self.store.get_user(user.id) or get_user_info(user)

        if not args:
            now = datetime.now(UTC)
            if user_info.tz:
                tz = user_info.tz
                local = now.astimezone(tz).strftime("%H:%M")
                await self._say(
                    ctx,
                    f"Your tz: `{user_info.tz.key}` (currently `{local} {fmt_offset(tz, now)}`)\n"
                    f"Use `/tz <IANA name>` to change, `/tz clear` to unset.",
                )
            else:
                await self._say(
                    ctx,
                    "Your tz: `unset (00?)`, defaulting to UTC.\n"
                    "Use `/tz <IANA name>` to set (e.g. `/tz America/New_York`).",
                )
            return

        arg = args[0].strip()
        if arg.lower() in ("clear", "reset", "default", "unset"):
            self.store.note_user(user_info)
            had = self.store.clear_user_tz(user.id)
            if had:
                await self._say(ctx, "Tz cleared (defaults to UTC).")
            else:
                await self._say(ctx, "No tz was set (currently defaulting to UTC).")
        else:
            if arg not in TIMEZONES:
                await self._say(
                    ctx,
                    f"Unknown tz `{arg}`. Use an IANA name like `America/New_York` or `Europe/London`.",
                )
            else:
                user_info.tz = ZoneInfo(arg)
                now = datetime.now(UTC)
                local = now.astimezone(user_info.tz).strftime("%H:%M")
                self.store.note_user(user_info)
                await self._say(
                    ctx,
                    f"Set your tz to `{arg}` (currently `{local} {fmt_offset(user_info.tz, now)}`).",
                )

    @command()
    async def command_help(self, ctx: CommandCtx):
        if ctx.is_private:
            ping_hint = "I'll respond to every message you send."
            admin_note = ""
        else:
            ping_hint = (
                f"Mention me with @{self.me.username} or reply to one of my messages "
                "to get a response. Other messages are just remembered for context."
            )
            admin_note = " _(admin-only in group chats)_"

        lines = [
            f"Hi! I'm {self.me.display_name}. {ping_hint}",
            "",
            "*Commands:*",
            f"`/start`: start listening here{admin_note}",
            f"`/stop`: pause listening here{admin_note}",
            "`/reset`: wipe this chat's history",
            "`/save`: back up unsaved messages to disk",
            f"`/model [name]`: show or set the model for this chat{admin_note}",
            f"`/load`: replace this chat's history with a Telegram Desktop `result.json`{admin_note}",
            "`/tz [name]`: show or set your timezone (IANA name, e.g. `America/New_York`)",
            "`/whoami`: show your telegram user id",
            "`/help`: show this message",
            "",
            "*Credentials:*",
            "`/setkey <key>`: store your own Anthropic key (DM only; I delete the message)",
            "`/forgetkey`: delete your stored key",
            "`/keystatus`: show your stored credential (masked)",
            f"`/allow <user_id|reply>`: add a user to the shared pool{admin_note}",
            f"`/disallow <user_id|reply>`: remove a user from the pool{admin_note}",
            f"`/poollist`: list pooled users{admin_note}",
            f"`/billing [triggering|designated]`: who pays in this group{admin_note}",
        ]
        await self._say(ctx, "\n".join(lines))

    @command(admin="in_groups")
    async def command_model(self, ctx: CommandCtx):
        chat_id = ctx.chat_id
        args = ctx.args

        if not args:
            pref = self.store.get_model_pref(chat_id)
            current = pref or self.config.default_claude_model
            source = "set for this chat" if pref else "default (no chat preference)"
            aliases = ", ".join(f"`{k}[x].[y] <-> claude-{v}-[x]-[y]`" for k, v in {"op": "opus", "s": "sonnet", "h": "haiku"}.items())
            models = ", ".join(f"`{m}`" for m in list(MODEL_ALIASES.keys()))
            await self._say(
                ctx,
                f"Current: `{current}` -- {source}\n"
                f"Aliases: {aliases}\n"
                f"Valid models: {models}\n"
                f"Use `/model <name>` to set, `/model clear` to revert.",
            )
            return

        arg = args[0].lower().strip()
        if arg in ("clear", "reset", "default", "unset"):
            had = self.store.clear_model_pref(chat_id)
            default = self.config.default_claude_model
            if had:
                await self._say(ctx, f"Preference cleared; reverted to default `{default}`")
            else:
                await self._say(ctx, f"No preference set; using default `{default}`")
            return

        resolved = MODEL_ALIASES.get(arg, arg)
        if resolved not in SUPPORTED_MODELS:
            await self._say(
                ctx,
                f"Unknown model `{resolved}`. Known: "
                + ", ".join(f"`{m}`" for m in sorted(SUPPORTED_MODELS)),
            )
            return

        self.store.set_model_pref(chat_id, resolved)
        logger.info(
            "Set model pref for chat %s to %s by user_id=%s",
            chat_id, resolved, ctx.user.id,
        )
        await self._say(ctx, f"Set to `{resolved}` for this chat")

    # ---- credential / pool / billing commands --------------------------
    # These interpolate user-controlled display names, so they send plain text
    # (no parse_mode) to avoid Markdown injection breaking the whole send.

    @staticmethod
    def _target_user_id(message: telegram.Message, args: list[str]) -> Optional[int]:
        """A target user id from a reply (preferred) or the first int-looking arg."""
        if message.reply_to_message and message.reply_to_message.from_user:
            return message.reply_to_message.from_user.id
        for a in args:
            try:
                return int(a)
            except ValueError:
                continue
        return None

    @command()
    async def command_setkey(self, ctx: CommandCtx):
        message = ctx.message

        # The inbound message carries the secret — delete it ASAP, any chat.
        deleted = False
        try:
            deleted = bool(await message.delete())
        except Exception:
            logger.warning("Could not delete /setkey message in chat %s", ctx.chat_id)

        if not ctx.is_private:
            tail = " I deleted it." if deleted else " Delete your message above — I couldn't."
            await self._say(ctx, f"Don't share a key in a group.{tail} DM me /setkey <key> instead.", markdown=False)
            return

        if not self.credentials.secrets.available:
            await self._say(ctx, "Credential storage isn't configured on this bot. Ask the admin to set CREDENTIAL_ENC_KEY.", markdown=False)
            return

        args = ctx.args
        if not args:
            await self._say(ctx, "Usage: /setkey <your-anthropic-key>. Send it here in this DM; I delete the message right away.", markdown=False)
            return

        key = args[0].strip()
        verdict = await self.credentials.validate_api_key(key)
        if verdict == "rejected":
            await self._say(ctx, "That key was rejected (bad or unauthorized). Nothing stored.", markdown=False)
            return

        self.credentials.set_credential(Credential(
            user_id=ctx.user.id,
            kind=CredentialKind.USER_API_KEY,
            secret=key,
            last_validated_at=datetime.now(UTC) if verdict == "ok" else None,
        ))
        tail = key[-4:] if len(key) >= 4 else "????"
        unverified = " (I couldn't fully validate it just now, but stored it)" if verdict == "unverified" else ""
        delnote = " Deleted your message." if deleted else " Couldn't delete your message — please remove it."
        await self._say(ctx, f"Stored your key (…{tail}); I'll bill your messages to it{unverified}.{delnote}", markdown=False)

    @command()
    async def command_forgetkey(self, ctx: CommandCtx):
        removed = self.credentials.forget_credential(ctx.user.id)
        if removed:
            await self._say(ctx, "Removed your stored credential.", markdown=False)
        else:
            await self._say(ctx, "You had no stored credential.", markdown=False)

    @command()
    async def command_keystatus(self, ctx: CommandCtx):
        cred = self.credentials.get_credential(ctx.user.id)
        pooled = self.credentials.is_pooled(ctx.user.id)
        if cred is not None:
            kind_label = {
                CredentialKind.USER_API_KEY: "API key",
                CredentialKind.OAUTH_SUBSCRIPTION: "subscription",
            }.get(cred.kind, cred.kind.value)
            validated = cred.last_validated_at.date().isoformat() if cred.last_validated_at else "not validated"
            pool_note = " (also in the shared pool, but your own key takes precedence)" if pooled else ""
            await self._say(
                ctx,
                f"Stored {kind_label} {cred.masked_secret()}, added {cred.created_at.date().isoformat()}, "
                f"last validated {validated}{pool_note}.",
                markdown=False,
            )
        elif pooled:
            await self._say(ctx, "No personal key — you're using the shared pool (bot's key).", markdown=False)
        else:
            await self._say(ctx, "No credential. DM me /setkey <key>, or ask an admin to add you to the pool.", markdown=False)

    @command(admin="always")
    async def command_allow(self, ctx: CommandCtx):
        target = self._target_user_id(ctx.message, ctx.args)
        if target is None:
            await self._say(ctx, "Usage: /allow <user_id> (or reply to one of their messages).", markdown=False)
            return
        added = self.credentials.add_to_pool(target)
        name = self.store.resolve_user(target).display_name
        if added:
            await self._say(ctx, f"Added {name} (id {target}) to the pool.", markdown=False)
        else:
            await self._say(ctx, f"{name} (id {target}) was already in the pool.", markdown=False)

    @command(admin="always")
    async def command_disallow(self, ctx: CommandCtx):
        target = self._target_user_id(ctx.message, ctx.args)
        if target is None:
            await self._say(ctx, "Usage: /disallow <user_id> (or reply to one of their messages).", markdown=False)
            return
        removed = self.credentials.remove_from_pool(target)
        name = self.store.resolve_user(target).display_name
        if removed:
            await self._say(ctx, f"Removed {name} (id {target}) from the pool.", markdown=False)
        else:
            await self._say(ctx, f"{name} (id {target}) wasn't in the pool.", markdown=False)

    @command(admin="always")
    async def command_poollist(self, ctx: CommandCtx):
        ids = self.credentials.list_pool()
        if not ids:
            await self._say(ctx, "The pool is empty. /allow <user_id> to add someone.", markdown=False)
            return
        lines = [f"Pool ({len(ids)}):"]
        lines += [f"- {self.store.resolve_user(uid).display_name} (id {uid})" for uid in ids]
        await self._say(ctx, "\n".join(lines), markdown=False)

    @command(admin="always")
    async def command_showprompt(self, ctx: CommandCtx):
        window = self.store.window(ctx.chat_id)
        base = self._base_prompt(
            ctx.chat_id, window.snapshot(), ctx.is_private,
            ctx.user.id if ctx.is_private else None,
        )
        rendered = self._render_changelog()
        if rendered:
            changelog_section = f"\n\n— Changelog —\n{rendered}"
        else:
            changelog_section = "\n\n— Changelog —\n(no entries yet)"
        await self._say(ctx, base + changelog_section, markdown=False)

    @command(admin="always")
    async def command_appendprompt(self, ctx: CommandCtx):
        if not ctx.args:
            await self._say(ctx, "Usage: /appendprompt [YYYY-MM-DD [HH:MM]] <text>", markdown=False)
            return

        words = list(ctx.args)
        timestamp, text_start = self._parse_date_prefix(words)
        text = " ".join(words[text_start:])

        if not text:
            await self._say(ctx, "Usage: /appendprompt [YYYY-MM-DD [HH:MM]] <text>", markdown=False)
            return

        entry = {"timestamp": timestamp, "text": text}
        self._prompt_changelog.append(entry)
        self._save_prompt_changelog()
        await self._say(ctx, f"Appended. Changelog is now:\n\n{self._render_changelog()}", markdown=False)

    def _parse_date_prefix(self, words: list[str]) -> tuple[str, int]:
        """Try to parse an optional YYYY-MM-DD [HH:MM] prefix from words.

        Returns (timestamp_str, index_of_first_text_word). If no date is
        found, timestamp is the current time and index is 0.
        """
        for fmt, n in [("%Y-%m-%d %H:%M", 2), ("%Y-%m-%d", 1)]:
            if len(words) >= n:
                try:
                    dt = datetime.strptime(" ".join(words[:n]), fmt)
                    return dt.strftime("%Y-%m-%d %H:%M +00"), n
                except ValueError:
                    pass
        return datetime.now(UTC).strftime("%Y-%m-%d %H:%M +00"), 0

    @command(admin="always")
    async def command_undoprompt(self, ctx: CommandCtx):
        if not self._prompt_changelog:
            await self._say(ctx, "Changelog is already empty.", markdown=False)
            return
        removed = self._prompt_changelog.pop()
        self._save_prompt_changelog()
        await self._say(ctx, f"Removed: [{removed['timestamp']}] {removed['text']}", markdown=False)


    @command(admin="in_groups")
    async def command_billing(self, ctx: CommandCtx):
        # DM short-circuit stays in-body: it REPLIES (not a silent drop), and
        # the admin="in_groups" gate already let the DM through unblocked.
        if ctx.is_private:
            await self._say(ctx, "Billing modes are for group chats. In a DM your own credential is always used.", markdown=False)
            return

        chat_id = ctx.chat_id
        args = ctx.args
        if not args:
            mode, designated = self.credentials.get_billing(chat_id)
            if mode == BillingMode.CHAT_DESIGNATED and designated is not None:
                name = self.store.resolve_user(designated).display_name
                desc = f"chat-designated — {name} (id {designated}) pays for everyone here"
            else:
                desc = "triggering-user — whoever pings pays (default)"
            await self._say(
                ctx,
                f"Billing here: {desc}.\n"
                "Change with /billing triggering, or /billing designated [reply|<user_id>] "
                "(with no reply/id, designates you).",
                markdown=False,
            )
            return

        sub = args[0].lower().strip()
        if sub in ("triggering", "trigger", "pinger", "triggering_user"):
            self.credentials.set_billing(chat_id, BillingMode.TRIGGERING_USER)
            await self._say(ctx, "Billing set: whoever pings pays (their own key or the pool).", markdown=False)
            return
        if sub in ("designated", "designate", "chat", "chat_designated"):
            if ctx.message.reply_to_message and ctx.message.reply_to_message.from_user:
                target = ctx.message.reply_to_message.from_user.id
            elif len(args) >= 2:
                try:
                    target = int(args[1])
                except ValueError:
                    await self._say(ctx, f"Couldn't parse a user id from {args[1]!r}.", markdown=False)
                    return
            else:
                target = ctx.user.id
            self.credentials.set_billing(chat_id, BillingMode.CHAT_DESIGNATED, target)
            name = self.store.resolve_user(target).display_name
            no_cred = self.credentials.get_credential(target) is None and not self.credentials.is_pooled(target)
            warn = (
                " Warning: they have no stored key and aren't in the pool, so pings here will be refused until that's fixed."
                if no_cred else ""
            )
            await self._say(ctx, f"Billing set: {name} (id {target}) pays for this chat.{warn}", markdown=False)
            return

        await self._say(ctx, f"Unknown option {sub!r}. Use 'triggering' or 'designated'.", markdown=False)

    def start(self):
        # TODO: maybe a pre-message handler to ingest metadata like user info, chat info, etc.
        # but this obscures the actual contract for data availability-- for now, prefer to have
        # clear separation of concerns and have each command manage its own needs.
        self.application.add_handler(CommandHandler("start", self.command_start), group=0)
        self.application.add_handler(CommandHandler("stop", self.command_stop), group=0)
        self.application.add_handler(CommandHandler("reset", self.command_reset), group=0)
        self.application.add_handler(CommandHandler("whoami", self.command_whoami), group=0)
        self.application.add_handler(CommandHandler("save", self.command_save), group=0)
        self.application.add_handler(CommandHandler("model", self.command_model), group=0)
        self.application.add_handler(CommandHandler("tz", self.command_tz), group=0)
        self.application.add_handler(CommandHandler("help", self.command_help), group=0)
        self.application.add_handler(CommandHandler("setkey", self.command_setkey), group=0)
        self.application.add_handler(CommandHandler("forgetkey", self.command_forgetkey), group=0)
        self.application.add_handler(CommandHandler("keystatus", self.command_keystatus), group=0)
        self.application.add_handler(CommandHandler("allow", self.command_allow), group=0)
        self.application.add_handler(CommandHandler("disallow", self.command_disallow), group=0)
        self.application.add_handler(CommandHandler("showprompt", self.command_showprompt), group=0)
        self.application.add_handler(CommandHandler("appendprompt", self.command_appendprompt), group=0)
        self.application.add_handler(CommandHandler("undoprompt", self.command_undoprompt), group=0)
        self.application.add_handler(CommandHandler("poollist", self.command_poollist), group=0)
        self.application.add_handler(CommandHandler("billing", self.command_billing), group=0)

        load_filter = (
            filters.Document.FileExtension("json")
            | filters.Document.MimeType("application/json")
        ) & filters.CaptionRegex(r"^/load(@\w+)?(\s|$)")
        self.application.add_handler(MessageHandler(load_filter, self.command_load), group=0)

        # Edited-message secret guard, registered ahead of on_message in group 1
        # (PTB runs the first matching handler per group). Its EDITED_MESSAGE
        # filter means normal messages fall through to on_message untouched,
        # while an edited-in key is caught and deleted. No ~COMMAND exclusion so
        # an edited-in `/setkey <key>` is caught too (CommandHandler ignores edits).
        edited_secret_filter = filters.UpdateType.EDITED_MESSAGE & (filters.TEXT | filters.CAPTION)
        self.application.add_handler(MessageHandler(edited_secret_filter, self.on_edited_message), group=1)

        # ~filters.COMMAND excludes text-commands; ~CaptionCommand() excludes the
        # caption equivalent (filters.COMMAND ignores caption_entities), so a
        # `/load`-captioned upload — already handled by command_load in group 0 —
        # doesn't also fall through here and get stored as the literal "/load".
        # Shared by on_message (group 1) and on_ping (group 2): neither should
        # see a caption-command.
        text_or_caption = (filters.TEXT | filters.CAPTION) & ~filters.COMMAND & ~CaptionCommand()
        self.application.add_handler(MessageHandler(text_or_caption, self.on_message), group=1)

        # on_ping gating is fully declarative: the chat must be active, the
        # message not ignore-prefixed, and either it's a DM (we answer
        # everything) or we were specifically addressed — @mentioned-by-our-
        # handle or a reply to one of our messages. The stateful filters
        # (ActiveChat/MentionsMe/RepliesToMe/NotIgnored) close over the store,
        # bot identity, and config, so the @incoming handler that follows can
        # assume anything reaching it has already qualified.
        #
        # NotIgnored is deliberately NOT applied to on_message: there the
        # secret guard must run before the ignore-prefix gate (so a key pasted
        # behind the prefix is still deleted), so that check stays in-body.
        addressed = (
            filters.ChatType.PRIVATE
            | MentionsMe(self.application)
            | RepliesToMe(self.application)
        )
        reply_filter = text_or_caption & ActiveChat(self.store) & NotIgnored(self.config) & addressed
        self.application.add_handler(MessageHandler(reply_filter, self.on_ping), group=2)

        logger.info("Starting bot polling")
        try:
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        finally:
            logger.info("Bot stopping; flushing persistence for %d active chat(s)", len(self.store.windows))
            self.store.persist_all()
            logger.info("Bot stopped")
