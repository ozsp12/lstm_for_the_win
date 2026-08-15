# LSTM for the Win

Reproducible continual-learning experiment for LSTM classification of product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

`train.csv` contains the current training corpus. `incoming.csv` simulates newly arrived reviews across five linguistic levels with variable length, slang, spelling noise, profanity, emojis, mixed sentiment, and controlled generation-to-generation variation. After each experiment, `goldtest=1` rows are promoted to the next training generation and the complete incoming batch is replaced.

Evaluation includes accuracy, precision, recall, macro/weighted F1, log-loss, Brier score, calibration error, segmented robustness metrics, and a TF-IDF logistic-regression baseline.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
lstm-pipeline train --run-id local --epochs 20
```

Python 3.12 · TensorFlow 2.20

The bundled data are synthetic and intended for controlled reproducibility experiments.
