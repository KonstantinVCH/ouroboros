"""QA Bot — configuration from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    token: str
    openrouter_key: str
    model: str
    admin_ids: set[int]
    price_stars: int

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("QA_BOT_TOKEN", "")
        if not token:
            raise RuntimeError("QA_BOT_TOKEN env var is required")

        openrouter_key = (
            os.environ.get("QA_BOT_OPENROUTER_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )

        model = os.environ.get(
            "QA_BOT_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
        )

        raw_admins = os.environ.get("QA_BOT_ADMIN_IDS", "")
        admin_ids: set[int] = set()
        for part in raw_admins.split(","):
            part = part.strip()
            if part.isdigit():
                admin_ids.add(int(part))

        price_stars = int(os.environ.get("QA_BOT_PRICE_STARS", "100"))

        return cls(
            token=token,
            openrouter_key=openrouter_key,
            model=model,
            admin_ids=admin_ids,
            price_stars=price_stars,
        )


# Singleton loaded lazily
_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config.from_env()
    return _cfg
