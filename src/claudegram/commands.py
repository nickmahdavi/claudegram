import telegram
from telegram.ext import Application, filters

from .config import Config
from .store import Store


# ---- stateful gating filters -------------------------------------------
# PTB's built-in filters are pure predicates over the message; these close
# over runtime state (the store, the bot identity) so the "should this handler
# run at all" decision lives declaratively in start() rather than imperatively
# in every handler. filter() runs synchronously on every update before
# dispatch, so each must stay cheap and non-blocking (in-memory only).

class ActiveChat(filters.MessageFilter):
    """The chat has been /start-ed (and not /stop-ped)."""

    def __init__(self, store: Store):
        super().__init__(name="ActiveChat")
        self.store = store

    def filter(self, message: telegram.Message) -> bool:
        return self.store.is_active(message.chat_id)


class MentionsMe(filters.MessageFilter):
    """A text/caption @mention whose handle is *this* bot's username.

    `filters.Entity(MENTION)` fires for any @handle; this narrows to ours by
    resolving each mention entity against the live bot username."""

    def __init__(self, app: Application):
        super().__init__(name="MentionsMe")
        self.app = app

    def filter(self, message: telegram.Message) -> bool:
        username = self.app.bot.username
        if not username:
            return False
        handle = f"@{username.lower()}"
        return any(
            e.type == telegram.MessageEntity.MENTION
            and message.parse_entity(e).lower() == handle
            for e in message.entities or []
        ) or any(
            e.type == telegram.MessageEntity.MENTION
            and message.parse_caption_entity(e).lower() == handle
            for e in message.caption_entities or []
        )


class RepliesToMe(filters.MessageFilter):
    """A reply to a message *this* bot sent (not any reply)."""

    def __init__(self, app: Application):
        super().__init__(name="RepliesToMe")
        self.app = app

    def filter(self, message: telegram.Message) -> bool:
        replied = message.reply_to_message
        return (
            replied is not None
            and replied.from_user is not None
            and replied.from_user.id == self.app.bot.id
        )


class NotIgnored(filters.MessageFilter):
    """Text/caption does NOT start with the configured ignore prefix.

    A silent drop is exactly right here: an ignore-prefixed message is meant to
    be invisible to the bot, so there's nothing to log or reply. Composed into
    the handler filters so the gate lives next to the others rather than as an
    in-body guard repeated per handler. No prefix configured => passes all."""

    def __init__(self, config: Config):
        super().__init__(name="NotIgnored")
        self.config = config

    def filter(self, message: telegram.Message) -> bool:
        prefix = self.config.ignore_prefix
        if not prefix:
            return True
        text = message.text or message.caption or ""
        return not text.startswith(prefix)
