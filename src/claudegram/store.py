import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from .message import Message, Window

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, data_dir: PathLike, log_dir: PathLike, input_budget: int):
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        self.input_budget = input_budget
        self.windows: dict[int, Window] = {}

        self._model_prefs: dict[int, str] = {}
        self._load_model_prefs()

        self._active_chats: set[int] = set()
        self._load_active_chats()

        self._user_prefs: dict[int, dict] = {}
        self._load_user_prefs()

        self._locks: dict[int, asyncio.Lock] = {}

        self._last_load_at: dict[int, datetime] = {}

    def __repr__(self) -> str:
        return (
            f"Store(data_dir={self.data_dir}, log_dir={self.log_dir}, "
            f"input_budget={self.input_budget}, "
            f"[{len(self.windows)} chats, {len(self._model_prefs)} model prefs, "
            f"{len(self._active_chats)} active, {len(self._user_prefs)} user prefs])"
        )

    def window(self, chat_id: int) -> Window:
        if chat_id not in self.windows:
            self.load_chat(chat_id)
        return self.windows[chat_id]

    def chat_path(self, chat_id: int) -> Path:
        return self.data_dir / f"chat_{chat_id}.jsonl"

    def context_path(self, chat_id: int) -> Path:
        return self.log_dir / f"chat_{chat_id}.context.log"

    @property
    def model_prefs_path(self) -> Path:
        return self.data_dir / "model_prefs.json"

    def get_model_pref(self, chat_id: int) -> Optional[str]:
        return self._model_prefs.get(chat_id)

    def set_model_pref(self, chat_id: int, model: str) -> None:
        self._model_prefs[chat_id] = model
        self._save_model_prefs()
        logger.info("Set model pref for chat %s: %s", chat_id, model)

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
            self._model_prefs = {int(k): v for k, v in raw.items()}
            logger.info("Loaded %d model pref(s) from %s", len(self._model_prefs), path)
        except Exception as e:
            logger.error("Failed to load model prefs from %s: %s — starting fresh", path, e, exc_info=True)
            self._model_prefs = {}

    def _save_model_prefs(self) -> None:
        path = self.model_prefs_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._model_prefs.items()}, f, indent=2)
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

    def get_user_tz(self, user_id: int) -> Optional[str]:
        return self._user_prefs.get(user_id, {}).get("tz")

    def set_user_tz(self, user_id: int, tz: str) -> None:
        prefs = self._user_prefs.setdefault(user_id, {})
        prefs["tz"] = tz
        self._save_user_prefs()
        logger.info("Set tz for user %s: %s", user_id, tz)

    def clear_user_tz(self, user_id: int) -> bool:
        prefs = self._user_prefs.get(user_id)
        if not prefs or "tz" not in prefs:
            return False
        prefs.pop("tz", None)
        if not prefs:
            self._user_prefs.pop(user_id, None)
        self._save_user_prefs()
        logger.info("Cleared tz for user %s", user_id)
        return True

    def _load_user_prefs(self) -> None:
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
                self._user_prefs[uid] = dict(v)
                loaded += 1
            except Exception as e:
                logger.warning("Skipping bad user pref entry %r: %s", k, e)
                skipped += 1
        logger.info("Loaded %d user pref(s) from %s (skipped %d)", loaded, path, skipped)

    def _save_user_prefs(self) -> None:
        path = self.users_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._user_prefs.items()}, f, indent=2)
        tmp.replace(path)

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
        return [self.persist(chat_id) for chat_id in self.windows]

    def lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def get_last_load_at(self, chat_id: int) -> Optional[datetime]:
        return self._last_load_at.get(chat_id)

    def mark_loaded(self, chat_id: int) -> None:
        self._last_load_at[chat_id] = datetime.now(timezone.utc)

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
        return backup

    def reset(self, chat_id: int) -> tuple[bool, bool, bool]:
        deleted_window = bool(self.windows.pop(chat_id, None))
        if deleted_window:
            logger.info("Reset chat history for chat %s", chat_id)

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