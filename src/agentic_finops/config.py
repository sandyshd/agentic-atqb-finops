from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv(override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    finops_env: str = os.getenv("FINOPS_ENV", "demo")
    ingest_interval_seconds: int = int(os.getenv("INGEST_INTERVAL_SECONDS", "2"))
    low_risk_confidence_threshold: float = float(
        os.getenv("LOW_RISK_CONFIDENCE_THRESHOLD", "0.75")
    )
    action_mode: str = os.getenv("ACTION_MODE", "guardrailed-auto")
    max_auto_monthly_savings_usd: float = float(
        os.getenv("MAX_AUTO_MONTHLY_SAVINGS_USD", "5000")
    )
    enable_cost_signal_simulation: bool = _env_bool("ENABLE_COST_SIGNAL_SIMULATION", True)
    enable_mag_simulation: bool = _env_bool("ENABLE_MAG_SIMULATION", True)
    enable_broker_baseline_quotes: bool = _env_bool("ENABLE_BROKER_BASELINE_QUOTES", False)
    broker_quote_refresh_seconds: int = int(os.getenv("BROKER_QUOTE_REFRESH_SECONDS", "45"))


settings = Settings()
