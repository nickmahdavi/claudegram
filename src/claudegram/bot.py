import asyncio
import logging
import time
from string import Template
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

import anthropic
import telegram
from telegram import Update, User
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import Config
from .error import ErrorClass, admin_failure_dm, admin_recovery_dm, classify_error, user_reply
from .identity import UserInfo
from .importer import parse_export
from .message import UTC, Message, Reply
from .model import MODEL_ALIASES, SUPPORTED_MODELS, PromptMode, complete, get_prompt
from .render import RenderMode, build_tz_directory, fmt_offset, render_history
from .store import Store

# Bot API caps document downloads at 20 MB
LOAD_MAX_BYTES = 18 * 1024 * 1024
# Discard pings older than this many seconds
STALE_PING_AGE_S = 60
TELEGRAM_CHAR_LIMIT = 4096
TIMEZONES: frozenset[str] = frozenset(available_timezones())

logger = logging.getLogger(__name__)

def get_user_info(user: User) -> UserInfo:
    return UserInfo(
        user_id=user.id,
        username=user.username or "",
        display_name=user.full_name,
    )

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
        PromptMode.CHAT: "You're an LLM in a group conversation. Messages from other participants are prefixed with their name + handle + UTC time + offset suffix (e.g. '14:32 +00'). You should just send your messages like normal (no prefix).\nYour display name is $display_name, and your username is $username. Users might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name.\n$user_tz_directory\nHave fun!",
        PromptMode.CHAT_PRIVATE: "You're an LLM in a private (1:1) conversation with $partner_display_name (@$partner_username). Their messages appear in human / user turns prefixed with their name + handle + local time + offset suffix (e.g. '14:32 -04'). Timestamps are rendered in their timezone ($partner_tz). You should just send your messages like normal (no prefix).\nYour display name is $display_name, and your username is $username. They might address you by your model name as well (e.g. Opus, Sonnet, version number, etc), so for context, your model name is $model_name. Have fun!",
    }

    def __init__(self, store: Store, config: Config):
        self.store = store
        self.config = config

        self._me: Optional[UserInfo] = None

        self.client = anthropic.AsyncClient(api_key=config.claude_api_key)
        self.application = Application.builder().token(self.config.telegram_bot_token).post_init(self._post_init).build()

        self.start_time = datetime.now(UTC)

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

    @property
    def me(self) -> UserInfo:
        if self._me is None:
            raise RuntimeError("Bot identity not loaded yet (app not initialized?)")
        return self._me

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        # why is from_user nullable at all?
        if not message or not message.from_user:
            return

        if not self.store.is_active(message.chat_id):
            return

        text = message.text or message.caption
        if not text:
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

        msg = Message(
            id=message.message_id,
            ts=message.date,
            user_id=message.from_user.id,
            text=text,
            reply_to=replied.message_id if replied else None,
            reply=reply,
        )

        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            evicted = window.append(msg)
            if evicted:
                logger.debug("Evicted %d message(s) from working set in chat %s", len(evicted), message.chat_id)
            self.store.persist(message.chat_id)

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
                    client=self.client,
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
                await self._handle_completion_error(message, exc)
                return

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
            if window is not None:
                self.store.persist(message.chat_id)

    def _is_admin(self, update: Update) -> bool:
        user = update.effective_user
        return user is not None and user.id in self.config.admin_user_ids

    async def _handle_completion_error(self, message: telegram.Message, exc: Exception) -> None:
        chat_id = message.chat_id
        err_class, err_desc = classify_error(exc)
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

        load_filter = (
            filters.Document.FileExtension("json")
            | filters.Document.MimeType("application/json")
        ) & filters.CaptionRegex(r"^/load(@\w+)?(\s|$)")
        self.application.add_handler(MessageHandler(load_filter, self.command_load), group=0)

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
