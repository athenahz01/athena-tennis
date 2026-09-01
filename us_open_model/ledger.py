from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from . import config

LEDGER = config.OUTPUTS / "prediction_ledger.jsonl"


def _append(row: dict[str, Any], path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def log_forecast(forecast: dict[str, Any], path: Path = LEDGER) -> str:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    context = forecast["context"]
    identity = "|".join(
        [
            forecast["model_version"],
            forecast["player1"]["name"],
            forecast["player2"]["name"],
            context["match_date"],
            created,
        ]
    )
    prediction_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    _append(
        {
            "event_type": "model_forecast",
            "prediction_id": prediction_id,
            "created_at": created,
            "forecast": forecast,
        },
        path,
    )
    return prediction_id


def log_market(
    prediction_id: str, market: dict[str, Any] | None, snapshot_path: Path | None,
    path: Path = LEDGER,
) -> None:
    _append(
        {
            "event_type": "market_observation",
            "prediction_id": prediction_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "snapshot_path": None if snapshot_path is None else str(snapshot_path),
            "market": market,
        },
        path,
    )

