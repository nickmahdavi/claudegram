
from dataclasses import dataclass
from datetime import datetime

import telegram
from telegram.ext import ContextTypes

from .identity import UserInfo


@dataclass
class CommandCtx:
    """A gated, projected command invocation — the command-world analog of
    Incoming. The @command decorator guarantees message/user are non-None and
    (if requested) the admin gate has passed before the body runs. update and
    context are kept as escape hatches for the few commands that need raw
    telegram objects (e.g. /load wants message.document, the credential
    commands want message.reply_to_message)."""
    message: telegram.Message
    user: telegram.User
    chat_id: int
    is_private: bool
    args: list[str]
    is_admin: bool
    update: telegram.Update
    context: ContextTypes.DEFAULT_TYPE


@dataclass
class Incoming:
    """A gated, parsed inbound message. By the time a handler receives one, the
    PTB filters have already decided it should run (active chat, addressed to us
    if a group); the handler just reads content."""
    sender: UserInfo
    message_id: int
    chat_id: int
    date: datetime
    text: str
    is_private: bool


@dataclass
class Outgoing:
    """A handler's reply. `system` marks bot-framework messages (refusals,
    errors) that get the system prefix and are not persisted to history."""
    text: str
    system: bool