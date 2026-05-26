"""Slightly questionable name."""

from dataclasses import dataclass
from typing import Optional, Self
from zoneinfo import ZoneInfo 

@dataclass(slots=True)
class UserInfo:
    user_id: int
    username: str
    display_name: str
    tz: Optional[ZoneInfo] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "tz": self.tz.key if isinstance(self.tz, ZoneInfo) else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            user_id=d["user_id"],
            username=d["username"],
            display_name=d["display_name"],
            tz=ZoneInfo(d["tz"]) if d.get("tz") else None,
        )
