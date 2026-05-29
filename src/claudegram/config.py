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

    @classmethod
    def load(cls) -> Self:
        return cls()  # type: ignore[call-arg]

    @field_validator("admin_user_ids", mode="before")
    @classmethod
    def _parse_admins(cls, v):
        if isinstance(v, str):
            return frozenset(int(x) for x in v.split(",") if x.strip())
        return v