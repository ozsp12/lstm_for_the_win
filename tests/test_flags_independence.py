from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


def test_emoji_assignment_is_not_sentiment_specific() -> None:
    config = SyntheticDataConfig(
        initial_train_rows=120,
        incoming_rows=120,
        emoji_fraction=0.50,
    )
    specs = SyntheticDataAgent(config)._specs(120, 0, True)
    for sentiment in ("positive", "neutral", "negative"):
        values = {spec["hasemoji"] for spec in specs if spec["sentiment"] == sentiment}
        assert values == {0, 1}
