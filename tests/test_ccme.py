import pandas as pd
from src.ccme import calculate_ccme_wqi

def test_ccme_range():
    df = pd.DataFrame({
        "ph": [7.0, 7.2],
        "do_mg_l": [6.0, 6.5],
        "dbo_mg_l": [2.0, 2.5],
        "ecoli_mpn_100ml": [100, 120],
        "total_n_mg_l": [0.8, 0.9],
        "total_p_mg_l": [0.05, 0.06],
        "turbidity_ntu": [30, 35],
    })
    r = calculate_ccme_wqi(df)
    assert 0 <= r["ccme_wqi"] <= 100
