from lstm_for_the_win.classification.pipeline import PipelineConfig


def test_default_text_capacity_tracks_richer_corpus() -> None:
    config = PipelineConfig(train_path="train.csv", incoming_path="incoming.csv", task="topic")
    assert config.max_tokens == 20_000
    assert config.sequence_length == 96
