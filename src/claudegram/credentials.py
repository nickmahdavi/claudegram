"""Per-user credential resolution: BYO API key, admin pool, and per-chat billing.

Replaces the single shared Anthropic key with **per-request credential
resolution**. Three kinds of credential:

  * ``USER_API_KEY``       — the user's own Anthropic key (stored encrypted).
  * ``OAUTH_SUBSCRIPTION`` — reserved for the Agent-SDK subscription path
                             (not wired yet; resolution/storage already work).
  * ``POOL``               — the user is allowed to use the bot's own key.

Persistence mirrors ``store.py`` exactly: atomic ``tmp.replace``, per-row
try/except that skips bad entries, lazy file creation. Secrets are only ever
written as Fernet ciphertext, in a ``0o600`` file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Optional, Union

import anthropic
from anthropic import AsyncClient

from .model import Model
from .secrets import Secrets

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Cheap model used to validate a freshly-supplied key with a 1-token call.
VALIDATION_MODEL = Model.HAIKU_4_5.value


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


class CredentialKind(StrEnum):
    USER_API_KEY = "user_api_key"
    OAUTH_SUBSCRIPTION = "oauth_subscription"  # reserved: Agent-SDK path, not wired yet
    POOL = "pool"  # uses the bot's own key


class BillingMode(StrEnum):
    TRIGGERING_USER = "triggering_user"  # default: whoever pinged pays
    CHAT_DESIGNATED = "chat_designated"  # one user's credential covers the chat


# Kinds backed by a user-owned secret (vs the shared pool key). Failures on
# these must be attributed to the user, never the bot's failure streak.
USER_OWNED_KINDS = frozenset({CredentialKind.USER_API_KEY, CredentialKind.OAUTH_SUBSCRIPTION})


@dataclass
class Credential:
    user_id: int
    kind: CredentialKind
    secret: Optional[str] = None  # plaintext, in memory only; None for POOL
    created_at: datetime = field(default_factory=_now)
    last_validated_at: Optional[datetime] = None

    def masked_secret(self) -> str:
        if not self.secret:
            return "(none)"
        tail = self.secret[-4:] if len(self.secret) >= 4 else "????"
        return f"…{tail}"


class CredentialStore:
    def __init__(self, data_dir: PathLike, secrets: Secrets, pool_api_key: str):
        self.data_dir = Path(data_dir)
        self.secrets = secrets

        # The shared bot key — used for POOL credentials and to validate user keys.
        self.pool_client = AsyncClient(api_key=pool_api_key, max_retries=3)

        # user_id -> Credential (USER_API_KEY / OAUTH_SUBSCRIPTION only; POOL is
        # implied by membership in self._pool, never stored here).
        self._creds: dict[int, Credential] = {}
        # user_ids allowed to use the pool (bot key).
        self._pool: set[int] = set()
        # chat_id -> (mode, designated_user_id)
        self._billing: dict[int, tuple[BillingMode, Optional[int]]] = {}

        # Client cache keyed by (user_id, kind, secret-fingerprint) so httpx
        # connection pools are reused across requests. Never build per request.
        self._clients: dict[tuple[int, str, str], AsyncClient] = {}

        self._load_credentials()
        self._load_pool()
        self._load_billing()

    # ---- paths ----------------------------------------------------------

    @property
    def credentials_path(self) -> Path:
        return self.data_dir / "credentials.json"

    @property
    def pool_path(self) -> Path:
        return self.data_dir / "pool_users.json"

    @property
    def billing_path(self) -> Path:
        return self.data_dir / "chat_billing.json"

    # ---- credentials (encrypted at rest) --------------------------------

    def _load_credentials(self) -> None:
        path = self.credentials_path
        if not path.exists():
            return
        if not self.secrets.available:
            logger.error(
                "credentials.json exists but CREDENTIAL_ENC_KEY is unavailable; "
                "stored keys cannot be decrypted and are ignored"
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error("Failed to load credentials from %s: %s — starting fresh", path, e, exc_info=True)
            return
        if not isinstance(raw, dict):
            logger.error("credentials file %s is not a JSON object; ignoring", path)
            return
        loaded = skipped = 0
        for k, v in raw.items():
            try:
                uid = int(k)
                if not isinstance(v, dict):
                    raise TypeError(f"expected dict, got {type(v).__name__}")
                kind = CredentialKind(v["kind"])
                secret = self.secrets.decrypt(v["enc_secret"])
                self._creds[uid] = Credential(
                    user_id=uid,
                    kind=kind,
                    secret=secret,
                    created_at=_parse_iso(v.get("created_at")) or _now(),
                    last_validated_at=_parse_iso(v.get("last_validated_at")),
                )
                loaded += 1
            except Exception as e:
                # Undecryptable (rotated key) or malformed rows are dropped; the
                # user re-enters their key. Never log the secret or ciphertext.
                logger.warning("Dropping unreadable credential row for %r: %s", k, type(e).__name__)
                skipped += 1
        logger.info("Loaded %d credential(s) from %s (skipped %d)", loaded, path, skipped)

    def _save_credentials(self) -> None:
        path = self.credentials_path
        out: dict[str, dict] = {}
        for uid, cred in self._creds.items():
            if cred.secret is None:
                continue
            out[str(uid)] = {
                "kind": cred.kind.value,
                "enc_secret": self.secrets.encrypt(cred.secret),
                "created_at": _iso(cred.created_at),
                "last_validated_at": _iso(cred.last_validated_at),
            }
        _atomic_write_secure(path, json.dumps(out, indent=2))

    def get_credential(self, user_id: int) -> Optional[Credential]:
        return self._creds.get(user_id)

    def set_credential(self, cred: Credential) -> None:
        if cred.secret is None or cred.kind not in USER_OWNED_KINDS:
            raise ValueError("set_credential requires a user-owned credential with a secret")
        self._creds[cred.user_id] = cred
        self._save_credentials()
        self._evict_user(cred.user_id)
        logger.info(
            "Stored %s credential for user %s (%s)",
            cred.kind.value, cred.user_id, cred.masked_secret(),
        )

    def forget_credential(self, user_id: int) -> bool:
        had = self._creds.pop(user_id, None) is not None
        if had:
            self._save_credentials()
            self._evict_user(user_id)
            logger.info("Forgot stored credential for user %s", user_id)
        return had

    # ---- pool -----------------------------------------------------------

    def _load_pool(self) -> None:
        path = self.pool_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._pool = {int(x) for x in raw}
            logger.info("Loaded %d pooled user(s) from %s", len(self._pool), path)
        except Exception as e:
            logger.error("Failed to load pool users from %s: %s — starting empty", path, e, exc_info=True)
            self._pool = set()

    def _save_pool(self) -> None:
        path = self.pool_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(self._pool), f, indent=2)
        tmp.replace(path)

    def is_pooled(self, user_id: int) -> bool:
        return user_id in self._pool

    def list_pool(self) -> list[int]:
        return sorted(self._pool)

    def add_to_pool(self, user_id: int) -> bool:
        if user_id in self._pool:
            return False
        self._pool.add(user_id)
        self._save_pool()
        logger.info("Added user %s to pool", user_id)
        return True

    def remove_from_pool(self, user_id: int) -> bool:
        if user_id not in self._pool:
            return False
        self._pool.discard(user_id)
        self._save_pool()
        logger.info("Removed user %s from pool", user_id)
        return True

    # ---- billing --------------------------------------------------------

    def _load_billing(self) -> None:
        path = self.billing_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error("Failed to load billing modes from %s: %s — starting fresh", path, e, exc_info=True)
            return
        if not isinstance(raw, dict):
            logger.error("billing file %s is not a JSON object; ignoring", path)
            return
        loaded = skipped = 0
        for k, v in raw.items():
            try:
                chat_id = int(k)
                mode = BillingMode(v["mode"])
                designated = v.get("designated_user_id")
                designated = int(designated) if designated is not None else None
                self._billing[chat_id] = (mode, designated)
                loaded += 1
            except Exception as e:
                logger.warning("Dropping bad billing entry for %r: %s", k, e)
                skipped += 1
        logger.info("Loaded %d billing mode(s) from %s (skipped %d)", loaded, path, skipped)

    def _save_billing(self) -> None:
        path = self.billing_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        out = {
            str(chat_id): {"mode": mode.value, "designated_user_id": designated}
            for chat_id, (mode, designated) in self._billing.items()
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        tmp.replace(path)

    def get_billing(self, chat_id: int) -> tuple[BillingMode, Optional[int]]:
        return self._billing.get(chat_id, (BillingMode.TRIGGERING_USER, None))

    def set_billing(self, chat_id: int, mode: BillingMode, designated_user_id: Optional[int] = None) -> None:
        if mode == BillingMode.TRIGGERING_USER:
            designated_user_id = None
        self._billing[chat_id] = (mode, designated_user_id)
        self._save_billing()
        logger.info("Set billing for chat %s: %s (designated=%s)", chat_id, mode.value, designated_user_id)

    # ---- resolution -----------------------------------------------------

    def resolve_credential(self, user_id: int, chat_id: int, is_private: bool) -> Optional[Credential]:
        """Return the Credential to bill this request to, or None to refuse.

        DM: resolve for the messaging user.
        Group: per the chat's billing mode — the triggering user (default) or
        the chat's designated user. None anywhere means "polite refusal".
        """
        if is_private:
            return self._resolve_for_user(user_id)
        mode, designated = self.get_billing(chat_id)
        if mode == BillingMode.CHAT_DESIGNATED:
            if designated is None:
                return None  # misconfigured: designated mode with no user
            return self._resolve_for_user(designated)
        return self._resolve_for_user(user_id)

    def _resolve_for_user(self, user_id: int) -> Optional[Credential]:
        # Precedence within a user: own credential (key/oauth) > pool > nothing.
        cred = self._creds.get(user_id)
        if cred is not None:
            return cred
        if user_id in self._pool:
            return Credential(user_id=user_id, kind=CredentialKind.POOL)
        return None

    # ---- client factory / cache ----------------------------------------

    def client_for(self, cred: Credential) -> AsyncClient:
        if cred.kind == CredentialKind.POOL:
            return self.pool_client
        if cred.kind == CredentialKind.USER_API_KEY:
            if not cred.secret:
                raise ValueError("USER_API_KEY credential has no secret")
            return self._cached_client(
                cred.user_id, cred.kind, cred.secret,
                lambda: AsyncClient(api_key=cred.secret, max_retries=3),
            )
        if cred.kind == CredentialKind.OAUTH_SUBSCRIPTION:
            # Routed through the Agent SDK, not messages.create — a separate
            # completion backend. Deliberately not wired in this phase.
            raise NotImplementedError("subscription (OAuth) completions are not wired yet")
        raise ValueError(f"unknown credential kind: {cred.kind}")

    @staticmethod
    def _fingerprint(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]

    def _cached_client(self, user_id: int, kind: CredentialKind, secret: str, factory) -> AsyncClient:
        fp = self._fingerprint(secret)
        key = (user_id, kind.value, fp)
        existing = self._clients.get(key)
        if existing is not None:
            return existing
        # A changed secret (rotated key) leaves a stale entry for this
        # (user, kind). Drop the reference — do NOT close it eagerly, since an
        # in-flight request on another coroutine may still be using it; GC and
        # aclose() at shutdown reclaim it.
        for stale in [k for k in self._clients if k[0] == user_id and k[1] == kind.value and k != key]:
            self._clients.pop(stale, None)
        client = factory()
        self._clients[key] = client
        return client

    def _evict_user(self, user_id: int) -> None:
        for key in [k for k in self._clients if k[0] == user_id]:
            self._clients.pop(key, None)

    # ---- validation -----------------------------------------------------

    async def validate_api_key(self, key: str) -> str:
        """Probe a key with a 1-token call. Returns one of:
        'ok'         — authenticated (incl. 402 out-of-credit: still a valid key),
        'rejected'   — definitively bad (auth/permission),
        'unverified' — couldn't tell (transient/network); caller may store anyway.
        Never logs or returns the key.
        """
        client = AsyncClient(api_key=key)
        try:
            await client.messages.create(
                model=VALIDATION_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return "ok"
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            return "rejected"
        except anthropic.APIStatusError as e:
            if e.status_code == 402:
                return "ok"  # valid key, just no credit balance
            return "unverified"
        except Exception:
            return "unverified"
        finally:
            try:
                await client.close()
            except Exception:
                pass

    # ---- lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        """Close every httpx pool we own. Called once at shutdown."""
        clients = list(self._clients.values()) + [self.pool_client]
        self._clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing client at shutdown", exc_info=True)


def _atomic_write_secure(path: Path, data: str) -> None:
    """Atomically write `data` to `path` with 0o600 perms.

    `os.open`'s mode only applies when the file is *created*, so a stale,
    looser-permissioned tmp file would otherwise slip through — we `fchmod`
    explicitly, and re-chmod the final path after the replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    os.chmod(path, 0o600)


__all__ = [
    "CredentialKind",
    "BillingMode",
    "Credential",
    "CredentialStore",
    "USER_OWNED_KINDS",
]
