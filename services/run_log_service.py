"""Local JSON logging for completed CareBridge workflow runs."""

import json
import os
from pathlib import Path
from typing import Any

from fastapi.encoders import jsonable_encoder


DEFAULT_RUN_LOG_DIR = Path(__file__).resolve().parents[1] / "run_logs"


def get_run_log_dir() -> Path:
    """Return the configured directory for local workflow logs."""

    configured = os.getenv("CAREBRIDGE_RUN_LOG_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_RUN_LOG_DIR


def write_run_log(run_id: str, record: dict[str, Any]) -> Path:
    """Atomically write one JSON file for a workflow run."""

    log_dir = get_run_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    destination = log_dir / f"{run_id}.json"
    temporary = log_dir / f".{run_id}.tmp"
    encoded = jsonable_encoder(record)

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(encoded, file, ensure_ascii=False, indent=2)
        file.write("\n")

    temporary.replace(destination)
    return destination
