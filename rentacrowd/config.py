"""Central configuration, loaded from the environment / .env once at import time."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunables in one place. Env vars win over these defaults."""

    model_config = SettingsConfigDict(env_prefix="RAC_", extra="ignore")

    # --- NIM connection (read without the RAC_ prefix) ---
    nvidia_api_key: str = Field("", alias="NVIDIA_API_KEY")
    nim_base_url: str = Field("https://integrate.api.nvidia.com/v1", alias="NIM_BASE_URL")

    # --- models ---
    panel_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    analysis_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    max_output_tokens: int = 2600

    # --- throughput guardrails ---
    requests_per_minute: int = 36
    max_concurrency: int = 4
    request_timeout: float = 180.0

    # --- panel sizing ---
    personas_per_segment: int = 6
    batch_size: int = 4

    # --- storage (inside the repo, so you can open the results) ---
    #   persona_library/ -> the reusable population (see personas/library.py)
    #   studies/         -> one folder per study run
    studies_dirname: str = "studies"

    @property
    def studies_dir(self) -> Path:
        p = REPO_ROOT / self.studies_dirname
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def requests_per_second(self) -> float:
        return self.requests_per_minute / 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
