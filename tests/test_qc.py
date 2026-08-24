import pandas as pd
from src.qc import apply_qc

def test_invalid_ph_is_excluded():
    df = pd.DataFrame({
        "station_id": ["E01"], "date": pd.to_datetime(["2026-01-01"]),
        "ph": [15.0], "do_mg_l": [5.0]
    })
    out = apply_qc(df)
    assert bool(out.loc[0, "qc_exclude"]) is True
    assert "ph:fora_limite_fisico" in out.loc[0, "qc_flags"]
