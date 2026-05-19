import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo, available_timezones

import telegram
from telegram import Update, User
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .importer import parse_export
from .logging_config import setup_logging
from .message import UTC, SYSTEM_PREFIX, Message, Reply, fmt_offset
from .model import Claude
from .store import Store

# Bot API caps document downloads at 20 MB
LOAD_MAX_BYTES = 18 * 1024 * 1024

# Discard pings older than
STALE_PING_AGE_S = 60

PathLike = Union[str, Path]
Identity = tuple[str, str]  # (username, display_name)

TIMEZONES: frozenset[str] = frozenset(available_timezones())

logger = logging.getLogger(__name__)


def get_identity(user: Optional[User]) -> Identity:
    if not user:
        return "anonymous", "anonymous"
    username = user.username or f"user_{user.id}"
    display_name = user.full_name or username
    return username, display_name


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
    def __init__(self, model: Claude, store: Store, app: Application,
        identity: Identity, ignore_prefix: Optional[str] = None, admin_ids: Optional[set[int]] = None):
        self.model = model
        self.store = store
        self.app = app
        self.username, self.display_name = identity
        self.ignore_prefix = ignore_prefix
        self.admin_ids = admin_ids or set()
        self.start_time = datetime.now(UTC)
        logger.info("Bot initialized: username=%s display_name=%s model=%r ignore_prefix=%r admins=%d start_time=%s", self.username, self.display_name, self.model, self.ignore_prefix, len(self.admin_ids), self.start_time.isoformat())

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        if not self.store.is_active(message.chat_id):
            return

        text = message.text or message.caption
        if not text:
            return

        if self.ignore_prefix and text.startswith(self.ignore_prefix):
            logger.debug("Skipping ignored message (prefix=%r) in chat %s", self.ignore_prefix, message.chat_id)
            return

        username, display_name = get_identity(message.from_user)

        reply = None
        replied = message.reply_to_message
        if replied:
            reply_username, reply_display = get_identity(replied.from_user)
            reply = Reply(
                username=reply_username,
                display_name=reply_display,
                text=message.quote.text if message.quote else replied.text or replied.caption or "",
                is_quote=message.quote is not None,
                ts=replied.date,
                user_id=replied.from_user.id if replied.from_user else None,
            )

        msg = Message(
            id=message.message_id,
            ts=message.date,
            username=username,
            display_name=display_name,
            text=text,
            reply_to=replied.message_id if replied else None,
            reply=reply,
            user_id=message.from_user.id if message.from_user else None,
        )

        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            evicted = window.append(msg)
            if evicted:
                logger.debug("Evicted %d message(s) from working set in chat %s", len(evicted), message.chat_id)
            self.store.persist(message.chat_id)

    async def on_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return

        if not self.store.is_active(message.chat_id):
            return

        text = message.text or message.caption
        if self.ignore_prefix and text and text.startswith(self.ignore_prefix):
            return

        is_private = message.chat.type == telegram.constants.ChatType.PRIVATE

        if is_private:
            trigger = "private"
        else:
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
            trigger = "mention" if mentioned else "reply"

        logger.info("on_ping triggered (%s) in chat %s", trigger, message.chat_id)

        now = datetime.now(UTC)
        cutoffs = [self.start_time, now - timedelta(seconds=STALE_PING_AGE_S)]
        last_load = self.store.get_last_load_at(message.chat_id)
        if last_load is not None:
            cutoffs.append(last_load)
        cutoff = max(cutoffs)
        if message.date < cutoff:
            age = (now - message.date).total_seconds()
            logger.info(
                "Skipping stale ping in chat %s (trigger=%s, age=%.1fs, cutoff=%s)",
                message.chat_id, trigger, age, cutoff.isoformat(),
            )
            return

        chat_model = self.store.get_model_pref(message.chat_id)
        partner = get_identity(message.from_user) if is_private else None
        partner_user_id = (
            message.from_user.id if (is_private and message.from_user) else None
        )

        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            window_snapshot = window.snapshot()
            window_tokens = window.tokens

        async with keep_typing(context.bot, message.chat_id):
            try:
                reply_text = await asyncio.to_thread(
                    self.model.complete,
                    window_snapshot,
                    window_tokens,
                    self.username,
                    self.display_name,
                    chat_model,
                    None, # fallback to default
                    is_private,
                    partner,
                    self.store.get_user_tz,
                    partner_user_id,
                )
            except Exception:
                logger.exception("Model error while completing for chat %s", message.chat_id)
                await message.reply_text(f"{SYSTEM_PREFIX} Hit an error, couldn't reply")
                return

        if not reply_text:
            logger.warning("Model returned empty reply for chat %s", message.chat_id)
            return

        chunks = [reply_text[i:i + 4096] for i in range(0, len(reply_text), 4096)]
        if len(chunks) > 1:
            logger.info(
                "Chunking reply for chat %s (%d chars -> %d pieces)",
                message.chat_id, len(reply_text), len(chunks),
            )
        sent = None
        for chunk in chunks:
            sent = await message.reply_text(chunk)
        if sent is None:
            logger.warning("No chunks sent for chat %s; skipping persist", message.chat_id)
            return

        bot_msg = Message(
            id=sent.message_id,
            ts=sent.date,
            username=self.username,
            display_name=self.display_name,
            text=reply_text,
            reply_to=message.message_id,
            reply=None,
            user_id=context.bot.id,
        )
        async with self.store.lock(message.chat_id):
            window = self.store.window(message.chat_id)
            window.append(bot_msg)
            self.store.persist(message.chat_id)

    def _is_admin(self, update: Update) -> bool:
        user = update.effective_user
        return user is not None and user.id in self.admin_ids

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
                f"{SYSTEM_PREFIX} Already listening here. `/stop` to pause.",
                parse_mode="Markdown",
            )
        else:
            if is_private:
                hint = "I'll respond to every message."
            else:
                hint = f"Mention me with @{self.username} or reply to my messages for a response."
            await message.reply_text(
                f"{SYSTEM_PREFIX} Hi! I'm {self.display_name}. {hint}\n"
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
                f"{SYSTEM_PREFIX} Going silent. `/start` to re-enable.",
                parse_mode="Markdown",
            )
            logger.info(
                "Deactivated chat %s by user_id=%s",
                chat_id,
                update.effective_user.id if update.effective_user else None,
            )
        else:
            await message.reply_text(
                f"{SYSTEM_PREFIX} Already inactive here.",
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
            await message.reply_text(f"{SYSTEM_PREFIX} History cleared.")
        else:
            logger.debug("Reset called on empty chat %s", message.chat_id)
            await message.reply_text(f"{SYSTEM_PREFIX} Nothing to clear.")
    
    async def command_whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        user = update.effective_user
        if not message or not user:
            return
        suffix = " (admin)" if user.id in self.admin_ids else ""
        await message.reply_text(f"{SYSTEM_PREFIX} Your telegram user id: `{user.id}`{suffix}", parse_mode="Markdown")

    async def command_save(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        async with self.store.lock(message.chat_id):
            n = self.store.persist(message.chat_id)
        if n:
            await message.reply_text(f"{SYSTEM_PREFIX} Saved {n} new message{"s" if n != 1 else ""}.")
        else:
            await message.reply_text(f"{SYSTEM_PREFIX} Nothing new to save.")

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
                f"{SYSTEM_PREFIX} Attach `result.json` from a Telegram Desktop "
                f"chat export with caption `/load`.",
                parse_mode="Markdown",
            )
            return

        filename = (document.file_name or "").lower()
        mime = document.mime_type or ""
        if not (filename.endswith(".json") or mime in ("application/json", "text/plain")):
            await message.reply_text(
                f"{SYSTEM_PREFIX} Expected a `.json` file (got `{document.file_name}`, mime `{mime}`).",
                parse_mode="Markdown",
            )
            return

        if document.file_size and document.file_size > LOAD_MAX_BYTES:
            await message.reply_text(
                f"{SYSTEM_PREFIX} File too large ({document.file_size:,} bytes; "
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
                    context.bot.id,
                    self.username,
                    self.display_name,
                )

                if result.kept == 0:
                    await message.reply_text(
                        f"{SYSTEM_PREFIX} Parsed {result.total} messages, kept 0 "
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
                await message.reply_text(f"{SYSTEM_PREFIX} Load failed: {e}")
                return
            finally:
                tmp_path.unlink(missing_ok=True)

        await message.reply_text(
            f"{SYSTEM_PREFIX} Loaded {result.kept}/{result.total} messages "
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

        args = context.args or []
        current = self.store.get_user_tz(user.id)

        if not args:
            now = datetime.now(UTC)
            if current:
                tz = ZoneInfo(current)
                local = now.astimezone(tz).strftime("%H:%M")
                await message.reply_text(
                    f"{SYSTEM_PREFIX} Your tz: `{current}` (currently `{local} {fmt_offset(tz, now)}`)\n"
                    f"Use `/tz <IANA name>` to change, `/tz clear` to unset.",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{SYSTEM_PREFIX} Your tz: `unset (00?)`, defaulting to UTC.\n"
                    "Use `/tz <IANA name>` to set (e.g. `/tz America/New_York`).",
                    parse_mode="Markdown",
                )
            return

        arg = args[0].strip()
        if arg.lower() in ("clear", "reset", "default", "unset"):
            had = self.store.clear_user_tz(user.id)
            if had:
                await message.reply_text(
                    f"{SYSTEM_PREFIX} Tz cleared; defaulting to UTC.",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{SYSTEM_PREFIX} No tz was set; defaulting to UTC.",
                    parse_mode="Markdown",
                )
            return

        if arg not in TIMEZONES:
            await message.reply_text(
                f"{SYSTEM_PREFIX} Unknown tz `{arg}`. Use an IANA name like `America/New_York` or `Europe/London`.",
                parse_mode="Markdown",
            )
            return

        self.store.set_user_tz(user.id, arg)
        tz = ZoneInfo(arg)
        now = datetime.now(UTC)
        local = now.astimezone(tz).strftime("%H:%M")
        await message.reply_text(
            f"{SYSTEM_PREFIX} Set your tz to `{arg}` (currently `{local} {fmt_offset(tz, now)}`).",
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
                f"Mention me with @{self.username} or reply to one of my messages "
                "to get a response. Other messages are just remembered for context."
            )
            admin_note = " _(admin-only in group chats)_"

        lines = [
            f"{SYSTEM_PREFIX} Hi! I'm {self.display_name}. {ping_hint}",
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
            current = pref or self.model.model
            source = "set for this chat" if pref else "default (no chat preference)"
            aliases = ", ".join(f"`{k}[x].[y] <-> claude-{v}-[x]-[y]`" for k, v in {"op": "opus", "s": "sonnet", "h": "haiku"}.items())
            models = ", ".join(f"`{m}`" for m in list(Claude.MODEL_ALIASES.keys()))
            await message.reply_text(
                f"{SYSTEM_PREFIX} Current: `{current}` -- {source}\n"
                f"Aliases: {aliases}\n"
                f"Valid models: {models}\n"
                f"Use `/model <name>` to set, `/model clear` to revert.",
                parse_mode="Markdown",
            )
            return

        arg = args[0].lower().strip()
        if arg in ("clear", "reset", "default", "unset"):
            had = self.store.clear_model_pref(chat_id)
            default = self.model.model
            if had:
                await message.reply_text(
                    f"{SYSTEM_PREFIX} Preference cleared; reverted to default `{default}`",
                    parse_mode="Markdown",
                )
            else:
                await message.reply_text(
                    f"{SYSTEM_PREFIX} No preference set; using default `{default}`",
                    parse_mode="Markdown",
                )
            return

        resolved = Claude.MODEL_ALIASES.get(arg, arg)
        if resolved not in self.model.KNOWN_MODELS:
            await message.reply_text(
                f"{SYSTEM_PREFIX} Unknown model `{resolved}`. Known: "
                + ", ".join(f"`{m}`" for m in sorted(self.model.KNOWN_MODELS)),
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
            f"{SYSTEM_PREFIX} Set to `{resolved}` for this chat",
            parse_mode="Markdown",
        )

    def start(self):
        self.app.add_handler(CommandHandler("start", self.command_start), group=0)
        self.app.add_handler(CommandHandler("stop", self.command_stop), group=0)
        self.app.add_handler(CommandHandler("reset", self.command_reset), group=0)
        self.app.add_handler(CommandHandler("whoami", self.command_whoami), group=0)
        self.app.add_handler(CommandHandler("save", self.command_save), group=0)
        self.app.add_handler(CommandHandler("model", self.command_model), group=0)
        self.app.add_handler(CommandHandler("tz", self.command_tz), group=0)
        self.app.add_handler(CommandHandler("help", self.command_help), group=0)

        load_filter = (
            filters.Document.FileExtension("json")
            | filters.Document.MimeType("application/json")
        ) & filters.CaptionRegex(r"^/load(@\w+)?(\s|$)")
        self.app.add_handler(MessageHandler(load_filter, self.command_load), group=0)

        text_or_caption = (filters.TEXT | filters.CAPTION) & ~filters.COMMAND
        self.app.add_handler(MessageHandler(text_or_caption, self.on_message), group=1)

        reply_filter = text_or_caption & (
            filters.ChatType.PRIVATE
            | filters.Entity(telegram.MessageEntity.MENTION)
            | filters.CaptionEntity(telegram.MessageEntity.MENTION)
            | filters.REPLY
        )
        self.app.add_handler(MessageHandler(reply_filter, self.on_ping), group=2)

        logger.info("Starting bot polling")
        try:
            self.app.run_polling(allowed_updates=Update.ALL_TYPES)
        finally:
            logger.info("Bot stopping; flushing persistence for %d active chat(s)", len(self.store.windows))
            self.store.persist_all()
            logger.info("Bot stopped")

def main() -> None:
    from .config import (
        CLAUDE_API_KEY,
        TELEGRAM_BOT_TOKEN,
        CLAUDE_MODEL,
        CLAUDE_BOT_USERNAME,
        CLAUDE_BOT_DISPLAY_NAME,
        TOKEN_BUDGET,
        REPLY_BUDGET,
        DATA_DIR,
        LOG_DIR,
        IGNORE_PREFIX,
        ADMIN_USER_IDS,
    )

    setup_logging(LOG_DIR)

    model = Claude(api_key=CLAUDE_API_KEY, model=CLAUDE_MODEL, max_tokens=REPLY_BUDGET)
    store = Store(data_dir=DATA_DIR, log_dir=LOG_DIR, input_budget=TOKEN_BUDGET)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot = Bot(
        model=model,
        store=store,
        app=app,
        identity=(
            CLAUDE_BOT_USERNAME,
            CLAUDE_BOT_DISPLAY_NAME,
        ),
        ignore_prefix=IGNORE_PREFIX,
        admin_ids=ADMIN_USER_IDS
    )
    bot.start()


if __name__ == "__main__":
    main()