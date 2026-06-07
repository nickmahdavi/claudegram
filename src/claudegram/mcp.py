"""MCP bearer-token manager for machine-to-machine (client_credentials) auth.

Fetches a token once, caches it, and transparently refreshes 60 seconds before
expiry. All callers share the same lock so concurrent requests during a refresh
don't trigger multiple token fetches.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Refresh this many seconds before the token actually expires, to avoid
# issuing a request with a token that expires mid-flight.
_REFRESH_BUFFER_S = 60


@dataclass
class _Token:
    access_token: str
    expires_at: float  # monotonic timestamp

    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at - _REFRESH_BUFFER_S


class McpTokenManager:
    """Async, concurrency-safe bearer-token manager for an MCP server."""

    def __init__(self, token_url: str, client_id: str, client_secret: str) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: Optional[_Token] = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> Optional[str]:
        """Return a valid bearer token, refreshing if necessary.

        Returns None if the token cannot be obtained, so callers can decide
        whether to proceed without MCP or surface an error.
        """
        async with self._lock:
            if self._token is None or self._token.is_expired():
                await self._refresh()
            return self._token.access_token if self._token else None

    async def _refresh(self) -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                expires_in = int(data.get("expires_in", 3600))
                self._token = _Token(
                    access_token=data["access_token"],
                    expires_at=time.monotonic() + expires_in,
                )
                logger.info("Obtained MCP access token (expires in %ds)", expires_in)
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to refresh MCP access token (HTTP %s %s); keeping existing token if valid",
                         exc.response.status_code, exc.response.reason_phrase)
        except Exception as exc:
            logger.error("Failed to refresh MCP access token (%s); keeping existing token if valid", type(exc).__name__)
