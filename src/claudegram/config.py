from typing import Annotated, Self

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .model import Model

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    claude_api_key: str
    telegram_bot_token: str
    default_claude_model: Model
    token_budget: int
    reply_budget: int
    data_dir: str = "data"
    log_dir: str = "logs"
    chat_view_log: bool = True
    ignore_prefix: str | None = None
    system_prefix: str = "<System>"
    admin_user_ids: Annotated[frozenset[int], NoDecode] = Field(default_factory=frozenset)
    # urlsafe-base64 32-byte Fernet key. Missing/malformed => credential storage
    # disabled (BYO-key/OAuth commands refuse, the admin pool still works).
    credential_enc_key: str | None = None

    # MCP server. All four must be set to enable MCP tool access.
    mcp_server_url: str | None = None    # full URL of the MCP endpoint
    mcp_token_url: str | None = None     # full URL of the OAuth token endpoint
    mcp_server_name: str | None = None   # name passed to the Anthropic API
    mcp_client_id: str | None = None
    mcp_client_secret: str | None = None
    mcp_system_prompt: str | None = None # appended to system prompt when MCP is active

    @property
    def mcp_enabled(self) -> bool:
        return bool(self.mcp_server_url and self.mcp_token_url and
                    self.mcp_server_name and self.mcp_client_id and self.mcp_client_secret)

    @classmethod
    def load(cls) -> Self:
        return cls()  # type: ignore[call-arg]

    @field_validator("admin_user_ids", mode="before")
    @classmethod
    def _parse_admins(cls, v):
        if isinstance(v, str):
            return frozenset(int(x) for x in v.split(",") if x.strip())
        return v