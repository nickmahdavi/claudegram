"""Thin symmetric-encryption wrapper for credentials at rest.

Design rule: if the master key is missing or malformed we **disable** credential
storage rather than crash the bot (`available == False`). The credential
commands then refuse politely and the admin pool (which stores no secrets) keeps
working. A rotated/changed key makes previously stored ciphertext undecryptable;
callers treat that as a dropped row (the user re-enters their key).
"""

from __future__ import annotations

import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class Secrets:
    def __init__(self, master_key: str | None):
        self._fernet: Fernet | None = None
        if not master_key:
            logger.warning(
                "CREDENTIAL_ENC_KEY not set; credential storage disabled "
                "(BYO-key commands will refuse, admin pool still works)"
            )
            return
        try:
            self._fernet = Fernet(master_key.encode("utf-8"))
        except Exception:
            # Never log the key itself, even malformed.
            logger.error(
                "CREDENTIAL_ENC_KEY is malformed (expected urlsafe-base64 32-byte "
                "Fernet key); credential storage disabled",
                exc_info=True,
            )
            self._fernet = None

    @property
    def available(self) -> bool:
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> str:
        if self._fernet is None:
            raise RuntimeError("Secrets unavailable: CREDENTIAL_ENC_KEY missing/malformed")
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Decrypt ciphertext. Raises `InvalidToken` if the token wasn't produced
        by this key (e.g. the key was rotated)."""
        if self._fernet is None:
            raise RuntimeError("Secrets unavailable: CREDENTIAL_ENC_KEY missing/malformed")
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")


__all__ = ["Secrets", "InvalidToken"]
