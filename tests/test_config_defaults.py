import json
from pathlib import Path


def test_production_config_enables_rich_variable_generation() -> None:
    config = json.loads(Path("config/synthetic_data.json").read_text(encoding="utf-8"))
    assert config["agent_version"] == "4.0.0"
    assert config["vary_counts"] is True
    assert config["initial_train_rows"] == 12000
    assert config["incoming_rows"] == 1800
    assert config["emoji_fraction"] > 0
    assert config["validation_fraction_jitter"] > 0
