from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SNAPSHOTS = DATA / "snapshots"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"

TML_BASE = "https://stats.tennismylife.org/data"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {
    "ATP": {
        "winner": "KXATPMATCH",
        "total_games": "KXATPGAMETOTAL",
        "exact_sets": "KXATPEXACTMATCH",
        "total_sets": "KXATPTOTALSETS",
    },
    "WTA": {
        "winner": "KXWTAMATCH",
        "total_games": "KXWTAGAMETOTAL",
        "exact_sets": "KXWTAEXACTMATCH",
        "total_sets": "KXWTATOTALSETS",
    },
}

MODEL_VERSION = "uso-hybrid-v1"
DEFAULT_START_YEAR = 2012
DEFAULT_TRAIN_CUTOFF = "2026-08-30"

for directory in (RAW, SNAPSHOTS, PROCESSED, ARTIFACTS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

