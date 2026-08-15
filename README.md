# LSTM for the Win

LSTM-based text-classification pipeline for product reviews, with independent models for sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Tasks

- Sentiment: `positive`, `neutral`, `negative`
- Topic: `smartphone`, `television`, `refrigerator`, `washing_machine`

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
lstm-pipeline train --run-id local --epochs 20
```

Python 3.12 · TensorFlow 2.20

The bundled data are synthetic and intended for reproducible software demonstration.
