from src.iqa import calculate_iqa, classify_iqa

def test_iqa_range_and_direction():
    good = {
        "do_saturation_pct": 98, "ecoli_mpn_100ml": 10, "ph": 7.2,
        "dbo_mg_l": 1.0, "delta_temp_c": 1.0, "total_n_mg_l": 0.3,
        "total_p_mg_l": 0.02, "turbidity_ntu": 5, "total_solids_mg_l": 40,
    }
    poor = {
        "do_saturation_pct": 45, "ecoli_mpn_100ml": 100000, "ph": 5.0,
        "dbo_mg_l": 15, "delta_temp_c": 15, "total_n_mg_l": 10,
        "total_p_mg_l": 1.0, "turbidity_ntu": 300, "total_solids_mg_l": 1200,
    }
    a = calculate_iqa(good)
    b = calculate_iqa(poor)
    assert 0 <= a <= 100
    assert 0 <= b <= 100
    assert a > b
    assert classify_iqa(a) in {"Boa", "Ótima"}
