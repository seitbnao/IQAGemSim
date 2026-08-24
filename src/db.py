from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "digital_twin.db"

def init_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scenario_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            station_id TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            scenario_json TEXT NOT NULL,
            baseline_iqa REAL,
            scenario_iqa REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            metrics_json TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def save_scenario(station_id, horizon, scenario, baseline_iqa, scenario_iqa, path: Path = DB_PATH):
    init_db(path)
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO scenario_runs(created_at,station_id,horizon,scenario_json,baseline_iqa,scenario_iqa) VALUES(?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            station_id,
            int(horizon),
            json.dumps(scenario, ensure_ascii=False),
            float(baseline_iqa),
            float(scenario_iqa),
        ),
    )
    con.commit()
    con.close()
