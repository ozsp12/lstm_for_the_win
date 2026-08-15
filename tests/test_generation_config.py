from lstm_for_the_win.agents import SyntheticDataConfig


def test_production_style_generation_ranges_are_seeded() -> None:
    config = SyntheticDataConfig(
        incoming_rows=1800,
        incoming_rows_jitter=300,
        vary_counts=True,
    )
    a = config.effective_generation(5)
    b = config.effective_generation(5)
    assert a == b
    assert 1500 <= int(a["incoming_rows"]) <= 2100
    assert int(a["incoming_rows"]) % 60 == 0
