from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests

from . import config

REQUIRED_COLUMNS = {
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
}

NUMERIC_COLUMNS = [
    "draw_size",
    "winner_ht",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_ht",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "best_of",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


def _source_name(tour: str, year: int) -> str:
    return f"{year}.csv" if tour.upper() == "ATP" else f"{year}_wta.csv"


def _local_name(tour: str, year: int) -> str:
    return f"{tour.lower()}_{year}.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, force: bool = False) -> dict:
    if destination.exists() and not force:
        return {"path": str(destination), "sha256": _sha256(destination), "cached": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "us-open-predictor/0.1 (research; public-data snapshot)"},
    )
    response.raise_for_status()
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_bytes(response.content)
    if len(response.content) < 100 or b"winner_name" not in response.content[:1000]:
        temp.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file failed schema sanity check: {url}")
    temp.replace(destination)
    return {"path": str(destination), "sha256": _sha256(destination), "cached": False}


def fetch_history(
    start_year: int = config.DEFAULT_START_YEAR,
    end_year: int | None = None,
    tours: tuple[str, ...] = ("ATP", "WTA"),
    *,
    force: bool = False,
) -> Path:
    end_year = end_year or dt.date.today().year
    manifest = {
        "provider": "TennisMyLife public data mirror",
        "base_url": config.TML_BASE,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": [],
    }
    for tour in tours:
        for year in range(start_year, end_year + 1):
            source = _source_name(tour, year)
            destination = config.RAW / _local_name(tour, year)
            item = download_file(f"{config.TML_BASE}/{source}", destination, force=force)
            item.update({"tour": tour, "year": year, "source": source})
            manifest["files"].append(item)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.SNAPSHOTS / f"tennis_data_manifest_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


ROUND_OFFSET = {
    "RR": 0,
    "R128": 0,
    "R64": 2,
    "R32": 4,
    "R16": 6,
    "QF": 8,
    "SF": 10,
    "F": 12,
    "BR": 12,
}


def _effective_date(row: pd.Series) -> pd.Timestamp:
    base = row["date"]
    round_name = str(row.get("round", ""))
    draw = row.get("draw_size")
    offset = ROUND_OFFSET.get(round_name, 0)
    # A 32-draw event begins at R32, so compress the generic Slam offsets.
    if pd.notna(draw) and float(draw) <= 64:
        start_offset = ROUND_OFFSET.get(f"R{int(draw)}", 0)
        offset = max(0, offset - start_offset)
    return base + pd.Timedelta(days=offset)


def is_completed_score(score: object) -> bool:
    text = str(score or "").upper()
    return bool(text and not re.search(r"\b(RET|W/O|WO|DEF|ABD|CANCELLED)\b", text))


def load_history(
    data_dir: Path = config.RAW,
    *,
    tours: tuple[str, ...] = ("ATP", "WTA"),
    cutoff: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tour in tours:
        for path in sorted(data_dir.glob(f"{tour.lower()}_*.csv")):
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            frame["tour"] = tour
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No ATP/WTA CSVs under {data_dir}. Run `us-open fetch` first."
        )
    matches = pd.concat(frames, ignore_index=True)
    matches["date"] = pd.to_datetime(
        matches["tourney_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    for column in NUMERIC_COLUMNS:
        if column in matches:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")
    matches = matches.dropna(subset=["date", "winner_name", "loser_name"])
    matches = matches[matches["score"].map(is_completed_score)].copy()
    matches["effective_date"] = matches.apply(_effective_date, axis=1)
    if cutoff:
        matches = matches[matches["effective_date"] <= pd.Timestamp(cutoff)]
    matches["match_key"] = (
        matches["tour"].astype(str)
        + ":"
        + matches["tourney_id"].astype(str)
        + ":"
        + matches["match_num"].astype(str)
    )
    matches = matches.drop_duplicates("match_key", keep="last")
    return matches.sort_values(
        ["effective_date", "tour", "tourney_id", "match_num"]
    ).reset_index(drop=True)

