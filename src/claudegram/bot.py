import asyncio
import logging
import os
import re
import time
from string import Template
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
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
    no_credential_reply,
    user_credential_failed_reply,
    user_reply,
)
from .identity import UserInfo
from .importer import parse_export
from .message import UTC, Forward, Message, Reply
from .model import MODEL_ALIASES, SUPPORTED_MODELS, PromptMode, complete, get_prompt
from .render import RenderMode, build_tz_directory, fmt_offset, render_history
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

class Bot:
    SYSTEM_PROMPTS = {
        PromptMode.PREFILL: "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with outputs of the commands.",
        PromptMode.CHAT: "You're an LLM in a group conversation. Messages from other participants are prefixed with their name + handle + UTC time + offset suffix (e.g. '14:32 +00'). You should just send your messages like normal (no prefix).\nSome messages carry extra context on the line(s) above the body: 're. <name (@handle) ...> text' means the sender is replying to that earlier message; '> <name ...> \"text\"' means they quoted a specific span of it; 'fwd. <name (@handle) ...>' means the message was forwarded and the tag is the *original* author, not the participant who reposted it. Forwards from hidden users or channels may omit the @handle or the timestamp.\nYour display name is $display_name, and your username is $username. Users might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name.\n$user_tz_directory\nHave fun!",
        PromptMode.CHAT_PRIVATE: "You're an LLM in a private (1:1) conversation with $partner_display_name (@$partner_username). Their messages appear in human / user turns prefixed with their name + handle + local time + offset suffix (e.g. '14:32 -04'). Timestamps are rendered in their timezone ($partner_tz). You should just send your messages like normal (no prefix).\nSome messages carry extra context on the line(s) above the body: 're. <name (@handle) ...> text' means they're replying to that earlier message; '> <name ...> \"text\"' means they quoted a specific span of it; 'fwd. <name (@handle) ...>' means the message was forwarded and the tag is the *original* author, not the person who sent it to you. Forwards from hidden users or channels may omit the @handle or the timestamp.\nYour display name is $display_name, and your username is $username. They might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name. Have fun!",
    }

    def __init__(self, store: Store, config: Config, credentials: CredentialStore):
        self.store = store
        self.config = config
        self.credentials = credentials

        self._me: Optional[UserInfo] = None

        # The shared bot/pool key lives in the credential store; keep a handle
        # for any direct use (e.g. legacy call sites). Per-request clients are
        # resolved via self.credentials.client_for(...).
        self.client = credentials.pool_client
        self.application = (
            Application.builder()
            .token(self.config.telegram_bot_token)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )

        self.start_time = datetime.now(UTC)

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

    def _view_system_prompt(self, chat_id: int, snapshot: list[Message], is_private: bool, partner_id: Optional[int]) -> str:
        """Rebuild the system prompt this chat would be sent right now, for the view
        log header. Mirrors on_ping's construction (mode, model pref, partner tz /
        group tz directory) so the logged 'System:' matches what the model sees."""
        prompt_mode = PromptMode.CHAT_PRIVATE if is_private else PromptMode.CHAT
        prompt_template = Template(self.SYSTEM_PROMPTS.get(prompt_mode, ""))
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

    async def on_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message or not message.from_user:
            return

        if not self.store.is_active(message.chat_id):
            return

        text = message.text or message.caption
        if not text:
            return
        if self.config.ignore_prefix and text and text.startswith(self.config.ignore_prefix):
            return
        
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE

        prompt_mode: PromptMode

        if is_private:
            prompt_mode = PromptMode.CHAT_PRIVATE
        else:
            prompt_mode = PromptMode.CHAT
            bot_handle = f"@{context.bot.username.lower()}"
            mentioned = (
                any(
                    e.type == telegram.MessageEntity.MENTION
                    and message.parse_entity(e).lower() == bot_handle
                    for e in message.entities or []
                )
                or any(
                    e.type == telegram.MessageEntity.MENTION
                    and message.parse_caption_entity(e).lower() == bot_handle
                    for e in message.caption_entities or []
                )
            )
            replied = message.reply_to_message
            replying_to_bot = (
                replied is not None
                and replied.from_user is not None
                and replied.from_user.id == context.bot.id
            )
            if not (mentioned or replying_to_bot):
                return

        render_mode: RenderMode = RenderMode.CHAT  # TODO Prefill is borked atm, will destub later

        logger.info("on_ping triggered in %s chat %s", message.chat.type, message.chat_id)

        # telegram buffers updates for 24h and for now I think the best design is to avoid spamming
        # users after the bot comes back up following an outage. but in the future we should buffer
        # these messages locally and allow the user to decide whether to process them or not,
        # probably in bulk with a single command. and this should also be unified with missed
        # buffered messages while out of credits, during anthropic API outages, etc. for now, since
        # we aren't handling or resending these, it's consistent to just drop stale messages
        now = datetime.now(UTC)
        cutoffs = [self.start_time, now - timedelta(seconds=STALE_PING_AGE_S)]
        last_load = self.store.get_last_load_at(message.chat_id)
        if last_load is not None:
            cutoffs.append(last_load)
        cutoff = max(cutoffs)
        if message.date < cutoff:
            age = (now - message.date).total_seconds()
            logger.info(
                "Skipping stale ping in chat %s (type=%s, age=%.1fs, cutoff=%s)",
                message.chat_id, message.chat.type, age, cutoff.isoformat(),
            )
            return

        # Resolve whose credential pays for this ping. None => polite refusal,
        # and crucially we must NOT touch the failure-streak/admin-alert
        # machinery (that tracks only the bot's pool key). Done before any
        # prompt building or typing indicator so a refusal is cheap and silent.
        cred = self.credentials.resolve_credential(message.from_user.id, message.chat_id, is_private)
        if cred is None:
            logger.info("No credential for user %s in chat %s; refusing", message.from_user.id, message.chat_id)
            await message.reply_text(f"{self.config.system_prefix} {no_credential_reply(is_private)}")
            return
        try:
            client = self.credentials.client_for(cred)
        except Exception:
            logger.exception(
                "Failed to build client for cred kind=%s user=%s", cred.kind.value, message.from_user.id,
            )
            await message.reply_text(f"{self.config.system_prefix} {credential_broken_reply()}")
            return

        prompt_template = Template(self.SYSTEM_PROMPTS.get(prompt_mode, ""))
        chat_model = self.store.get_model_pref(message.chat_id) or self.config.default_claude_model
        partner: Optional[UserInfo] = None
        display_tz: Optional[ZoneInfo] = UTC
        tz_directory: Optional[str] = None

        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            snapshot = window.snapshot()
            window_tokens = window.tokens
            known_users = window.known_users()
            incarnation = self.store.incarnation(message.chat_id)

        if prompt_mode == PromptMode.CHAT_PRIVATE:
            # might be weird defaulting to anonymous in dms?
            # maybe we should add an unknown user distinct from anon
            partner = self.store.resolve_user(message.from_user.id)
            display_tz = partner.tz
        else:
            known_users.discard(self.me.user_id)  # don't show me in my own directory
            tz_directory = build_tz_directory(known_users, self.store.resolve_user)

        system = get_prompt(
            prompt_template=prompt_template,
            model=chat_model,
            bot_info=self.me,
            partner=partner,
            tz_directory=tz_directory
        )

        messages = render_history(snapshot, self.me, render_mode, self.store.resolve_user, display_tz)

        async with keep_typing(context.bot, message.chat_id):
            try:
                reply = await complete(
                    client=client,
                    model=chat_model,
                    system=system,
                    messages=messages,
                    max_tokens=self.config.reply_budget
                )
                # window_tokens is len // 4 + 5 + (overhead) per message. Empirically the ratio
                # here is about 1.05, but it's different for opus 4.7 and likely later
                # models, so good to keep it around
                ratio = reply.true_input / window_tokens if window_tokens else float("nan")
                logger.info(
                    "Token usage for %s: estimated=%d actual=%d (ratio=%.2f) "\
                    "[uncached=%d cache_write=%d cache_read=%d output=%d]",
                    chat_model, window_tokens, reply.true_input, ratio,
                    reply.input_tokens, reply.cache_write, reply.cache_read, reply.output_tokens,
                )
            except Exception as exc:
                await self._handle_completion_error(message, exc, cred)
                return

        # Only a POOL (bot-key) success clears the chat's failure streak.
        # Clearing it on a user-key success would let one BYO user mask an
        # ongoing pool outage and flap the recovery/outage admin alerts.
        prior_failures = 0
        if cred.kind == CredentialKind.POOL:
            async with self.store.lock(message.chat_id):
                prior_failures = self.store.note_success(message.chat_id)
        if prior_failures:
            await self._notify_admins_recovered(message.chat_id, prior_failures)

        if not reply.text:
            logger.warning("Model returned empty reply for chat %s", message.chat_id)
            return

        chunks = [reply.text[i:i + TELEGRAM_CHAR_LIMIT] for i in range(0, len(reply.text), TELEGRAM_CHAR_LIMIT)]
        if len(chunks) > 1:
            logger.info(
                "Chunking reply for chat %s (%d chars -> %d pieces)",
                message.chat_id, len(reply.text), len(chunks),
            )
        async with self.store.lock(message.chat_id):
            # A concurrent /reset or /load while we were awaiting the model
            # would have wiped or replaced this chat's history. If its
            # incarnation moved since we snapshotted, the reply we computed is
            # for a conversation that no longer exists: still send it (the user
            # pinged and deserves an answer) but don't persist it onto the new
            # window. We hold the lock across the whole send loop, so
            # the incarnation can't change again mid-loop.
            # 
            # This is currently bullshit in 99% of cases
            stale = self.store.incarnation(message.chat_id) != incarnation
            if stale:
                logger.warning(
                    "Chat %s history changed during completion (incarnation %d -> %d); "
                    "sending reply without persisting",
                    message.chat_id, incarnation, self.store.incarnation(message.chat_id),
                )
            window = None if stale else self.store.window(message.chat_id)
            for i, chunk in enumerate(chunks):
                first = i == 0
                sent = await message.reply_text(chunk)
                if sent is None or sent.text is None:
                    logger.warning("Chunk %d not sent for chat %s; skipping persist", i, message.chat_id)
                elif window is not None:
                    window.append(Message(
                        id=sent.message_id,
                        ts=sent.date,
                        user_id=self.me.user_id,
                        text=sent.text,
                        reply_to=message.message_id if first else None,
                        reply= Reply(
                            user_id=message.from_user.id,
                            text=message.text or message.caption or "",
                            is_quote=False,
                            ts=message.date
                        ) if first else None
                    ))
            view_snapshot = None
            if window is not None:
                self.store.persist(message.chat_id)
                if self._should_write_view(message.chat_id, force=True):
                    view_snapshot = window.snapshot()

        if view_snapshot is not None:
            await self._write_chat_view(message.chat_id, view_snapshot, display_tz, is_private, message.from_user.id)

    def _is_admin(self, update: Update) -> bool:
        user = update.effective_user
        return user is not None and user.id in self.config.admin_user_ids

    async def _handle_completion_error(
        self, message: telegram.Message, exc: Exception, cred: Credential,
    ) -> None:
        chat_id = message.chat_id
        err_class, err_desc = classify_error(exc)

        # Per-user failure isolation. A failure on a user's OWN credential
        # (bad/expired key, out-of-credit, their account rate-limited, etc.)
        # must be attributed to that user and must NOT feed the chat's
        # failure-streak / admin-alert machinery, which tracks only the bot's
        # pool key. Out-of-credit (402) is the most common BYO failure and the
        # easiest to misattribute as "the bot is down".
        if cred.kind in USER_OWNED_KINDS:
            logger.warning(
                "User-owned credential failure (user=%s, kind=%s, class=%s, desc=%s)",
                message.from_user.id if message.from_user else None,
                cred.kind.value, err_class.value, err_desc,
            )
            try:
                await message.reply_text(
                    f"{self.config.system_prefix} {user_credential_failed_reply(err_class)}"
                )
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
                    await message.reply_text(text)
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

    async def command_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        chat_id = message.chat_id
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if not is_private and not self._is_admin(update):
            logger.warning(
                "Non-admin /start denied in group chat %s (user_id=%s)",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )
            return

        was_active = self.store.set_active(chat_id, True)
        if was_active:
            await message.reply_text(
                f"{self.config.system_prefix} Already listening here. `/stop` to pause.",
                parse_mode="Markdown",
            )
        else:
            if is_private:
                hint = "I'll respond to every message."
            else:
                hint = f"Mention me with @{self.me.username} or reply to my messages for a response."
            await message.reply_text(
                f"{self.config.system_prefix} Hi! I'm {self.me.display_name}. {hint}\n"
                f"`/stop` to stop listening, `/help` for commands.",
                parse_mode="Markdown",
            )
            logger.info(
                "Activated chat %s by user_id=%s",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )

    async def command_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        chat_id = message.chat_id
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if not is_private and not self._is_admin(update):
            logger.warning(
                "Non-admin /stop denied in group chat %s (user_id=%s)",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )
            return

        was_active = self.store.set_active(chat_id, False)
        if was_active:
            await message.reply_text(
                f"{self.config.system_prefix} Going silent. `/start` to re-enable.",
                parse_mode="Markdown",
            )
            logger.info(
                "Deactivated chat %s by user_id=%s",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )
        else:
            await message.reply_text(
                f"{self.config.system_prefix} Already inactive here.",
                parse_mode="Markdown",
            )

    async def command_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        async with self.store.lock(message.chat_id):
            window, file, ctx = self.store.reset(message.chat_id)

        if window or file or ctx:
            logger.info("Reset chat %s (window=%s, file=%s, ctx=%s)", message.chat_id, window, file, ctx)
            await message.reply_text(f"{self.config.system_prefix} History cleared.")
        else:
            logger.debug("Reset called on empty chat %s", message.chat_id)
            await message.reply_text(f"{self.config.system_prefix} Nothing to clear.")
    
    async def command_whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        if not message or not user:
            return
        suffix = " (admin)" if user.id in self.config.admin_user_ids else ""
        await message.reply_text(f"{self.config.system_prefix} Your telegram user id: `{user.id}`{suffix}", parse_mode="Markdown")

    async def command_save(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        async with self.store.lock(message.chat_id):
            n = self.store.persist(message.chat_id)
        if n:
            await message.reply_text(f"{self.config.system_prefix} Saved {n} new message{"s" if n != 1 else ""}.")
        else:
            await message.reply_text(f"{self.config.system_prefix} Nothing new to save.")

    async def command_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        chat_id = message.chat_id
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if not is_private and not self._is_admin(update):
            logger.warning(
                "Non-admin /load denied in group chat %s (user_id=%s)",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )
            return

        document = message.document
        if document is None:
            await message.reply_text(
                f"{self.config.system_prefix} Attach `result.json` from a Telegram Desktop "
                f"chat export with caption `/load`.",
                parse_mode="Markdown",
            )
            return

        filename = (document.file_name or "").lower()
        mime = document.mime_type or ""
        if not (filename.endswith(".json") or mime in ("application/json", "text/plain")):
            await message.reply_text(
                f"{self.config.system_prefix} Expected a `.json` file (got `{document.file_name}`, mime `{mime}`).",
                parse_mode="Markdown",
            )
            return

        if document.file_size and document.file_size > LOAD_MAX_BYTES:
            await message.reply_text(
                f"{self.config.system_prefix} File too large ({document.file_size:,} bytes; "
                f"limit is {LOAD_MAX_BYTES:,})."
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
                    await message.reply_text(
                        f"{self.config.system_prefix} Parsed {result.total} messages, kept 0 "
                        f"(dropped: {result.dropped_system} system, "
                        f"{result.dropped_commands} commands, "
                        f"{result.dropped_service} service, "
                        f"{result.dropped_non_text} non-text). "
                        f"Refusing to replace history with nothing.",
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
                await message.reply_text(f"{self.config.system_prefix} Load failed: {e}")
                return
            finally:
                tmp_path.unlink(missing_ok=True)

        await message.reply_text(
            f"{self.config.system_prefix} Loaded {result.kept}/{result.total} messages "
            f"(dropped: {result.dropped_system} system, "
            f"{result.dropped_commands} commands, "
            f"{result.dropped_service} service, "
            f"{result.dropped_non_text} non-text). "
            f"Backup: `{backup.name}`",
            parse_mode="Markdown",
        )

    async def command_tz(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        if not message or not user:
            return
        if user.is_bot:
            await message.reply_text(f"{self.config.system_prefix} I can't set a timezone for a bot/anonymous user. Try starting a private chat with me.")
            return

        args = context.args or []
        user_info = self.store.get_user(user.id) or get_user_info(user)

        if not args:
            now = datetime.now(UTC)
            if user_info.tz:
                tz = user_info.tz
                local = now.astimezone(tz).strftime("%H:%M")
                await message.reply_text(
                    f"{self.config.system_prefix} Your tz: `{user_info.tz.key}` (currently `{local} {fmt_offset(tz, now)}`)\n"
                    f"Use `/tz <IANA name>` to change, `/tz clear` to unset.",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{self.config.system_prefix} Your tz: `unset (00?)`, defaulting to UTC.\n"
                    "Use `/tz <IANA name>` to set (e.g. `/tz America/New_York`).",
                    parse_mode="Markdown",
                )
            return

        arg = args[0].strip()
        if arg.lower() in ("clear", "reset", "default", "unset"):
            self.store.note_user(user_info)
            had = self.store.clear_user_tz(user.id)
            if had:
                await message.reply_text(
                    f"{self.config.system_prefix} Tz cleared (defaults to UTC).",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{self.config.system_prefix} No tz was set (currently defaulting to UTC).",
                    parse_mode="Markdown",
                )
        else:
            if arg not in TIMEZONES:
                await message.reply_text(
                    f"{self.config.system_prefix} Unknown tz `{arg}`. Use an IANA name like `America/New_York` or `Europe/London`.",
                    parse_mode="Markdown",
                )
            else:
                user_info.tz = ZoneInfo(arg)
                now = datetime.now(UTC)
                local = now.astimezone(user_info.tz).strftime("%H:%M")
                self.store.note_user(user_info)
                await message.reply_text(
                    f"{self.config.system_prefix} Set your tz to `{arg}` (currently `{local} {fmt_offset(user_info.tz, now)}`).",
                    parse_mode="Markdown",
                )

    async def command_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if is_private:
            ping_hint = "I'll respond to every message you send."
            admin_note = ""
        else:
            ping_hint = (
                f"Mention me with @{self.me.username} or reply to one of my messages "
                "to get a response. Other messages are just remembered for context."
            )
            admin_note = " _(admin-only in group chats)_"

        lines = [
            f"{self.config.system_prefix} Hi! I'm {self.me.display_name}. {ping_hint}",
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
        await message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def command_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if not is_private and not self._is_admin(update):
            logger.warning(
                "Non-admin /model denied in group chat %s (user_id=%s)",
                message.chat_id,
                update.effective_user.id if update.effective_user else None,
            )
            return # don't leak

        chat_id = message.chat_id
        args = context.args or []

        if not args:
            pref = self.store.get_model_pref(chat_id)
            current = pref or self.config.default_claude_model
            source = "set for this chat" if pref else "default (no chat preference)"
            aliases = ", ".join(f"`{k}[x].[y] <-> claude-{v}-[x]-[y]`" for k, v in {"op": "opus", "s": "sonnet", "h": "haiku"}.items())
            models = ", ".join(f"`{m}`" for m in list(MODEL_ALIASES.keys()))
            await message.reply_text(
                f"{self.config.system_prefix} Current: `{current}` -- {source}\n"
                f"Aliases: {aliases}\n"
                f"Valid models: {models}\n"
                f"Use `/model <name>` to set, `/model clear` to revert.",
                parse_mode="Markdown",
            )
            return

        arg = args[0].lower().strip()
        if arg in ("clear", "reset", "default", "unset"):
            had = self.store.clear_model_pref(chat_id)
            default = self.config.default_claude_model
            if had:
                await message.reply_text(
                    f"{self.config.system_prefix} Preference cleared; reverted to default `{default}`",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{self.config.system_prefix} No preference set; using default `{default}`",
                    parse_mode="Markdown",
                )
            return

        resolved = MODEL_ALIASES.get(arg, arg)
        if resolved not in SUPPORTED_MODELS:
            await message.reply_text(
                f"{self.config.system_prefix} Unknown model `{resolved}`. Known: "
                + ", ".join(f"`{m}`" for m in sorted(SUPPORTED_MODELS)),
                parse_mode="Markdown",
            )
            return

        self.store.set_model_pref(chat_id, resolved)
        logger.info(
            "Set model pref for chat %s to %s by user_id=%s",
            chat_id,
            resolved,
            update.effective_user.id if update.effective_user else None,
        )
        await message.reply_text(
            f"{self.config.system_prefix} Set to `{resolved}` for this chat",
            parse_mode="Markdown",
        )

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

    async def command_setkey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message or not message.from_user:
            return
        prefix = self.config.system_prefix
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE

        # The inbound message carries the secret — delete it ASAP, any chat.
        deleted = False
        try:
            deleted = bool(await message.delete())
        except Exception:
            logger.warning("Could not delete /setkey message in chat %s", message.chat_id)

        if not is_private:
            tail = " I deleted it." if deleted else " Delete your message above — I couldn't."
            await message.reply_text(
                f"{prefix} Don't share a key in a group.{tail} DM me /setkey <key> instead."
            )
            return

        if not self.credentials.secrets.available:
            await message.reply_text(
                f"{prefix} Credential storage isn't configured on this bot. Ask the admin to set CREDENTIAL_ENC_KEY."
            )
            return

        args = context.args or []
        if not args:
            await message.reply_text(
                f"{prefix} Usage: /setkey <your-anthropic-key>. Send it here in this DM; I delete the message right away."
            )
            return

        key = args[0].strip()
        verdict = await self.credentials.validate_api_key(key)
        if verdict == "rejected":
            await message.reply_text(f"{prefix} That key was rejected (bad or unauthorized). Nothing stored.")
            return

        self.credentials.set_credential(Credential(
            user_id=message.from_user.id,
            kind=CredentialKind.USER_API_KEY,
            secret=key,
            last_validated_at=datetime.now(UTC) if verdict == "ok" else None,
        ))
        tail = key[-4:] if len(key) >= 4 else "????"
        unverified = " (I couldn't fully validate it just now, but stored it)" if verdict == "unverified" else ""
        delnote = " Deleted your message." if deleted else " Couldn't delete your message — please remove it."
        await message.reply_text(
            f"{prefix} Stored your key (…{tail}); I'll bill your messages to it{unverified}.{delnote}"
        )

    async def command_forgetkey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        if not message or not user:
            return
        prefix = self.config.system_prefix
        removed = self.credentials.forget_credential(user.id)
        if removed:
            await message.reply_text(f"{prefix} Removed your stored credential.")
        else:
            await message.reply_text(f"{prefix} You had no stored credential.")

    async def command_keystatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        if not message or not user:
            return
        prefix = self.config.system_prefix
        cred = self.credentials.get_credential(user.id)
        pooled = self.credentials.is_pooled(user.id)
        if cred is not None:
            kind_label = {
                CredentialKind.USER_API_KEY: "API key",
                CredentialKind.OAUTH_SUBSCRIPTION: "subscription",
            }.get(cred.kind, cred.kind.value)
            validated = cred.last_validated_at.date().isoformat() if cred.last_validated_at else "not validated"
            pool_note = " (also in the shared pool, but your own key takes precedence)" if pooled else ""
            await message.reply_text(
                f"{prefix} Stored {kind_label} {cred.masked_secret()}, added {cred.created_at.date().isoformat()}, "
                f"last validated {validated}{pool_note}."
            )
        elif pooled:
            await message.reply_text(f"{prefix} No personal key — you're using the shared pool (bot's key).")
        else:
            await message.reply_text(
                f"{prefix} No credential. DM me /setkey <key>, or ask an admin to add you to the pool."
            )

    async def command_allow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        if not self._is_admin(update):
            logger.warning("Non-admin /allow denied (user_id=%s)", update.effective_user.id if update.effective_user else None)
            return
        prefix = self.config.system_prefix
        target = self._target_user_id(message, context.args or [])
        if target is None:
            await message.reply_text(f"{prefix} Usage: /allow <user_id> (or reply to one of their messages).")
            return
        added = self.credentials.add_to_pool(target)
        name = self.store.resolve_user(target).display_name
        if added:
            await message.reply_text(f"{prefix} Added {name} (id {target}) to the pool.")
        else:
            await message.reply_text(f"{prefix} {name} (id {target}) was already in the pool.")

    async def command_disallow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        if not self._is_admin(update):
            logger.warning("Non-admin /disallow denied (user_id=%s)", update.effective_user.id if update.effective_user else None)
            return
        prefix = self.config.system_prefix
        target = self._target_user_id(message, context.args or [])
        if target is None:
            await message.reply_text(f"{prefix} Usage: /disallow <user_id> (or reply to one of their messages).")
            return
        removed = self.credentials.remove_from_pool(target)
        name = self.store.resolve_user(target).display_name
        if removed:
            await message.reply_text(f"{prefix} Removed {name} (id {target}) from the pool.")
        else:
            await message.reply_text(f"{prefix} {name} (id {target}) wasn't in the pool.")

    async def command_poollist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        if not self._is_admin(update):
            logger.warning("Non-admin /poollist denied (user_id=%s)", update.effective_user.id if update.effective_user else None)
            return
        prefix = self.config.system_prefix
        ids = self.credentials.list_pool()
        if not ids:
            await message.reply_text(f"{prefix} The pool is empty. /allow <user_id> to add someone.")
            return
        lines = [f"{prefix} Pool ({len(ids)}):"]
        lines += [f"- {self.store.resolve_user(uid).display_name} (id {uid})" for uid in ids]
        await message.reply_text("\n".join(lines))

    async def command_billing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        prefix = self.config.system_prefix
        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE
        if is_private:
            await message.reply_text(
                f"{prefix} Billing modes are for group chats. In a DM your own credential is always used."
            )
            return
        if not self._is_admin(update):
            logger.warning("Non-admin /billing denied in chat %s", message.chat_id)
            return

        chat_id = message.chat_id
        args = context.args or []
        if not args:
            mode, designated = self.credentials.get_billing(chat_id)
            if mode == BillingMode.CHAT_DESIGNATED and designated is not None:
                name = self.store.resolve_user(designated).display_name
                desc = f"chat-designated — {name} (id {designated}) pays for everyone here"
            else:
                desc = "triggering-user — whoever pings pays (default)"
            await message.reply_text(
                f"{prefix} Billing here: {desc}.\n"
                "Change with /billing triggering, or /billing designated [reply|<user_id>] "
                "(with no reply/id, designates you)."
            )
            return

        sub = args[0].lower().strip()
        if sub in ("triggering", "trigger", "pinger", "triggering_user"):
            self.credentials.set_billing(chat_id, BillingMode.TRIGGERING_USER)
            await message.reply_text(f"{prefix} Billing set: whoever pings pays (their own key or the pool).")
            return
        if sub in ("designated", "designate", "chat", "chat_designated"):
            if message.reply_to_message and message.reply_to_message.from_user:
                target = message.reply_to_message.from_user.id
            elif len(args) >= 2:
                try:
                    target = int(args[1])
                except ValueError:
                    await message.reply_text(f"{prefix} Couldn't parse a user id from {args[1]!r}.")
                    return
            else:
                target = update.effective_user.id if update.effective_user else None
            if target is None:
                await message.reply_text(f"{prefix} Usage: /billing designated [reply|<user_id>] (defaults to you).")
                return
            self.credentials.set_billing(chat_id, BillingMode.CHAT_DESIGNATED, target)
            name = self.store.resolve_user(target).display_name
            no_cred = self.credentials.get_credential(target) is None and not self.credentials.is_pooled(target)
            warn = (
                " Warning: they have no stored key and aren't in the pool, so pings here will be refused until that's fixed."
                if no_cred else ""
            )
            await message.reply_text(f"{prefix} Billing set: {name} (id {target}) pays for this chat.{warn}")
            return

        await message.reply_text(f"{prefix} Unknown option {sub!r}. Use 'triggering' or 'designated'.")

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

        text_or_caption = (filters.TEXT | filters.CAPTION) & ~filters.COMMAND
        self.application.add_handler(MessageHandler(text_or_caption, self.on_message), group=1)

        reply_filter = text_or_caption & (
            filters.ChatType.PRIVATE
            | filters.Entity(telegram.MessageEntity.MENTION)
            | filters.CaptionEntity(telegram.MessageEntity.MENTION)
            | filters.REPLY
        )
        self.application.add_handler(MessageHandler(reply_filter, self.on_ping), group=2)

        logger.info("Starting bot polling")
        try:
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        finally:
            logger.info("Bot stopping; flushing persistence for %d active chat(s)", len(self.store.windows))
            self.store.persist_all()
            logger.info("Bot stopped")
