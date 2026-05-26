import asyncio
import json
import logging
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from .identity import UserInfo
from .model import Model
from .message import Message, Window

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, data_dir: PathLike, log_dir: PathLike, input_budget: int):
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        self.input_budget = input_budget
        self.windows: dict[int, Window] = {}

        self._model_prefs: dict[int, Model] = {}
        self._load_model_prefs()

        self._active_chats: set[int] = set()
        self._load_active_chats()

        self._users: dict[int, UserInfo] = {}
        self._load_users()

        self._locks: dict[int, asyncio.Lock] = {}

        self._last_load_at: dict[int, datetime] = {}

        self._failure_counts: dict[int, int] = {}
        self._last_error_reply_at: dict[int, datetime] = {}
        self._admin_alerted: dict[int, bool] = {}

        # Bumped whenever a chat's window is wiped (reset) or wholesale replaced (load),
        # i.e. on a discontinuity in history. Slow in-flight completions should detect that the
        # conversation it was answering no longer exists before persisting.
        self._incarnation: dict[int, int] = {}

    def __repr__(self) -> str:
        return (
            f"Store(data_dir={self.data_dir}, log_dir={self.log_dir}, "
            f"input_budget={self.input_budget}, "
            f"[{len(self.windows)} chats, {len(self._model_prefs)} model prefs, "
            f"{len(self._active_chats)} active, {len(self._users)} users])"
        )

    def window(self, chat_id: int) -> Window:
        if chat_id not in self.windows:
            self.load_chat(chat_id)
        return self.windows[chat_id]

    def chat_path(self, chat_id: int) -> Path:
        return self.data_dir / f"chat_{chat_id}.jsonl"

    def view_path(self, chat_id: int) -> Path:
        # Human-readable "as the bot sees it" render, lives in log_dir (it's a log, not data).
        return self.log_dir / f"chat_{chat_id}.view.log"

    def context_path(self, chat_id: int) -> Path:
        return self.log_dir / f"chat_{chat_id}.context.log"

    @property
    def model_prefs_path(self) -> Path:
        return self.data_dir / "model_prefs.json"

    def get_model_pref(self, chat_id: int) -> Optional[Model]:
        # _model_prefs only ever holds valid Models: normalization happens in
        # set_model_pref and bad on-disk entries are dropped in _load_model_prefs.
        return self._model_prefs.get(chat_id)

    def set_model_pref(self, chat_id: int, model: Model | str) -> None:
        resolved = model if isinstance(model, Model) else Model(model)  # raises on garbage
        self._model_prefs[chat_id] = resolved
        self._save_model_prefs()
        logger.info("Set model pref for chat %s: %s", chat_id, resolved.value)

    def clear_model_pref(self, chat_id: int) -> bool:
        had = chat_id in self._model_prefs
        self._model_prefs.pop(chat_id, None)
        if had:
            self._save_model_prefs()
            logger.info("Cleared model pref for chat %s", chat_id)
        return had

    def _load_model_prefs(self) -> None:
        path = self.model_prefs_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error("Failed to load model prefs from %s: %s — starting fresh", path, e, exc_info=True)
            self._model_prefs = {}
            return
        parsed: dict[int, Model] = {}
        for k, v in raw.items():
            try:
                parsed[int(k)] = Model(v)
            except (ValueError, TypeError) as e:
                logger.warning("Dropping invalid model pref for chat %s: %r (%s)", k, v, e)
        self._model_prefs = parsed
        logger.info("Loaded %d model pref(s) from %s", len(self._model_prefs), path)

    def _save_model_prefs(self) -> None:
        path = self.model_prefs_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v.value for k, v in self._model_prefs.items()}, f, indent=2)
        tmp.replace(path)

    @property
    def active_chats_path(self) -> Path:
        return self.data_dir / "active_chats.json"

    def is_active(self, chat_id: int) -> bool:
        return chat_id in self._active_chats

    def set_active(self, chat_id: int, active: bool) -> bool:
        was = chat_id in self._active_chats
        if active:
            self._active_chats.add(chat_id)
        else:
            self._active_chats.discard(chat_id)
        if was != active:
            self._save_active_chats()
            logger.info("Chat %s %s", chat_id, "activated" if active else "deactivated")
        return was

    def _load_active_chats(self) -> None:
        path = self.active_chats_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._active_chats = {int(x) for x in raw}
            logger.info("Loaded %d active chat(s) from %s", len(self._active_chats), path)
        except Exception as e:
            logger.error("Failed to load active chats from %s: %s — starting empty", path, e, exc_info=True)
            self._active_chats = set()

    def _save_active_chats(self) -> None:
        path = self.active_chats_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(str(c) for c in self._active_chats), f, indent=2)
        tmp.replace(path)

    @property
    def users_path(self) -> Path:
        return self.data_dir / "users.json"

    def _load_users(self) -> None:
        path = self.users_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error("Failed to load user prefs from %s: %s — starting fresh", path, e, exc_info=True)
            return
        if not isinstance(raw, dict):
            logger.error("user prefs file %s is not a JSON object (got %s); ignoring",
                         path, type(raw).__name__)
            return
        loaded = 0
        skipped = 0
        for k, v in raw.items():
            try:
                uid = int(k)
                if not isinstance(v, dict):
                    raise TypeError(f"expected dict, got {type(v).__name__}")
                self._users[uid] = UserInfo.from_dict(v)
                loaded += 1
            except Exception as e:
                logger.warning("Skipping bad user pref entry %r: %s", k, e)
                skipped += 1
        logger.info("Loaded %d user pref(s) from %s (skipped %d)", loaded, path, skipped)

    def _save_users(self) -> None:
        path = self.users_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v.to_dict() for k, v in self._users.items()}, f, indent=2)
        tmp.replace(path)

    def clear_user_tz(self, user_id: int) -> bool:
        user = self._users.get(user_id)
        if not user or not user.tz:
            return False
        user.tz = None
        self._save_users()
        logger.info("Cleared tz for user %s", user_id)
        return True
    
    def get_user(self, user_id: int) -> Optional[UserInfo]:
        user_info = self._users.get(user_id)
        return replace(user_info) if user_info else None
    
    def resolve_user(self, user_id: int) -> UserInfo:
        user_info = self.get_user(user_id)
        if user_info:
            return user_info
        return UserInfo(user_id=user_id, username=f"user_{user_id}", display_name=f"User {user_id}", tz=None)

    def set_user(self, user_info: UserInfo):
        # Unceremonious.
        self._users[user_info.user_id] = replace(user_info)
        self._save_users()
        logger.debug("Set info for user %s: %s", user_info.user_id, user_info)
    
    def note_user(self, user_info: UserInfo) -> bool:
        existing = self._users.get(user_info.user_id)
        if existing is None:
            self._users[user_info.user_id] = replace(user_info)
            self._save_users()
            logger.debug("Recorded new user %s: %s", user_info.user_id, user_info)
            return True

        # Only fields the caller actually supplied (truthy) can overwrite; never
        # clobber stored info with blanks. tz unset goes through clear_user_tz.
        changes = {
            field: new
            for field, new, old in (
                ("username", user_info.username, existing.username),
                ("display_name", user_info.display_name, existing.display_name),
                ("tz", user_info.tz, existing.tz),
            )
            if new and new != old
        }
        if not changes:
            logger.debug("No changes to persist for user %s", user_info.user_id)
            return False

        for field, new in changes.items():
            setattr(existing, field, new)
        self._save_users()
        logger.debug("Updated user %s: %s", user_info.user_id, changes)
        return True

    def load_chat(self, chat_id: int) -> Window:
        path = self.chat_path(chat_id)
        try:
            window = Window.from_file(path, budget=self.input_budget)
        except Exception as e:
            logger.error("Failed to load chat history for chat %s from %s: %s",
                chat_id, path, e, exc_info=True
            )
            window = Window(budget=self.input_budget)
        self.windows[chat_id] = window
        if window.size:
            logger.info("Loaded chat %s: %d messages (%d working, %d tokens)",
                        chat_id, window.size, len(window), window.tokens)
        else:
            logger.debug("Starting fresh window for chat %s", chat_id)
        return window
    
    def persist(self, chat_id: int) -> int:
        window = self.windows.get(chat_id)
        if window is None:
            return 0
        n = window.write(self.chat_path(chat_id))
        if n:
            logger.debug("Persisted %d new message(s) for chat %s", n, chat_id)
        return n
    
    def persist_all(self) -> list[int]:
        # list() so a concurrent load_chat/reset mutating self.windows can't
        # raise "dictionary changed size during iteration" at shutdown.
        return [self.persist(chat_id) for chat_id in list(self.windows)]

    def lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def incarnation(self, chat_id: int) -> int:
        """Opaque marker of the chat's current continuous history. Bumped only
        on a discontinuity -- reset()/replace_chat() -- never on appends.
        Snapshot it before a long await and re-check afterwards: if it changed,
        the history was wiped/replaced out from under you."""
        return self._incarnation.get(chat_id, 0)

    def get_last_load_at(self, chat_id: int) -> Optional[datetime]:
        return self._last_load_at.get(chat_id)

    def mark_loaded(self, chat_id: int) -> None:
        self._last_load_at[chat_id] = datetime.now(timezone.utc)

    # ---- API-failure tracking -------------------------------------------

    # After this many consecutive failures in a chat, DM the admins. Once.
    ADMIN_ALERT_THRESHOLD: int = 3
    # Cap on the exponential backoff between user-facing error replies (sec).
    # 30 min means we'll still occasionally surface "still broken" to whoever
    # was pinging the chat, but not multiple times a minute during an outage.
    ERROR_REPLY_BACKOFF_CAP_S: int = 1800

    def get_failure_count(self, chat_id: int) -> int:
        return self._failure_counts.get(chat_id, 0)

    def note_failure(self, chat_id: int) -> int:
        """Increment the per-chat consecutive-failure counter. Returns the
        new count. Callers should follow up with `should_send_error_reply`
        and `should_alert_admin` to decide what to do about it."""
        self._failure_counts[chat_id] = self._failure_counts.get(chat_id, 0) + 1
        return self._failure_counts[chat_id]

    def note_success(self, chat_id: int) -> int:
        """Mark a successful completion in this chat. Clears all failure
        state. Returns the count of failures we'd accumulated IF the streak
        had reached admin-alert territory (so callers can announce recovery
        to admins); zero otherwise.
        """
        count = self._failure_counts.pop(chat_id, 0)
        self._last_error_reply_at.pop(chat_id, None)
        was_alerted = self._admin_alerted.pop(chat_id, False)
        return count if was_alerted else 0

    def should_send_error_reply(self, chat_id: int) -> bool:
        """Exponential-backoff gate on user-facing "I'm broken" replies.

        First failure: always reply. Subsequent failures: reply only if it's
        been at least 2^(count-1) seconds since the last error reply we sent
        in this chat, capped at ERROR_REPLY_BACKOFF_CAP_S. Cooldown is on
        the OUTBOUND reply, not the API call (the SDK does its own retries
        internally).
        """
        count = self._failure_counts.get(chat_id, 0)
        if count <= 1:
            return True
        last = self._last_error_reply_at.get(chat_id)
        if last is None:
            # We have a streak but haven't replied yet in it (e.g., the
            # initial replies were rate-limited away). Speak up.
            return True
        cooldown_s = min(2 ** (count - 1), self.ERROR_REPLY_BACKOFF_CAP_S)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= cooldown_s

    def mark_error_reply_sent(self, chat_id: int) -> None:
        """Record that we just sent a user-facing error reply, so the next
        `should_send_error_reply` call uses this as the cooldown anchor."""
        self._last_error_reply_at[chat_id] = datetime.now(timezone.utc)

    def should_alert_admin(self, chat_id: int) -> bool:
        """True iff this chat has hit ADMIN_ALERT_THRESHOLD consecutive
        failures AND we haven't already alerted on this streak."""
        if self._admin_alerted.get(chat_id):
            return False
        return self._failure_counts.get(chat_id, 0) >= self.ADMIN_ALERT_THRESHOLD

    def mark_admin_alerted(self, chat_id: int) -> None:
        """Record that we just DM'd the admins about this chat's failure
        streak. Cleared on the next `note_success`."""
        self._admin_alerted[chat_id] = True

    def replace_chat(self, chat_id: int, messages: list[Message]) -> Path:
        path = self.chat_path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        backup = path.with_suffix(path.suffix + f".bak.{time.time_ns()}")
        if path.exists():
            os.replace(path, backup)
            logger.info("Backed up chat %s history to %s", chat_id, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg.to_dict()) + "\n")
        os.replace(tmp, path)
        logger.info("Replaced chat %s history with %d message(s)", chat_id, len(messages))

        self.windows.pop(chat_id, None)
        self._incarnation[chat_id] = self._incarnation.get(chat_id, 0) + 1
        return backup

    def reset(self, chat_id: int) -> tuple[bool, bool, bool]:
        deleted_window = bool(self.windows.pop(chat_id, None))
        if deleted_window:
            logger.info("Reset chat history for chat %s", chat_id)
        self._incarnation[chat_id] = self._incarnation.get(chat_id, 0) + 1

        path = self.chat_path(chat_id)
        chat_exists = path.exists()
        if chat_exists:
            path.unlink()
            logger.info("Deleted chat history (%s) for chat %s", path, chat_id)

        ctx_path = self.context_path(chat_id)
        ctx_exists = ctx_path.exists()
        if ctx_exists:
            ctx_path.unlink()
            logger.info("Deleted context history (%s) for chat %s", ctx_path, chat_id)
        
        return (deleted_window, chat_exists, ctx_exists)
