# LSTM for the Win

Reproducible continual-learning experiment for Long Short-Term Memory (LSTM) classification of synthetic product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Data lifecycle

`data/input/train.csv` is the cumulative training corpus. `data/input/incoming.csv` contains the current unseen synthetic batch across five linguistic levels, with variable length, slang, spelling noise, profanity, emojis, and mixed sentiment. After an evaluation, only rows marked `goldtest=1` are promoted to the training corpus; the next incoming batch is then generated. Existing training rows are not automatically deleted when the text generator version changes.

Workflow runs are incremental. Each run receives its own timestamped directory under `data/output/`, and previous run directories are retained. `data/output/latest.json` is only a pointer to the newest completed run.

## Analysis artifacts

Each new run persists only two analytical artifacts:

- `analysis.json`: canonical source for the live dashboard and all downstream analysis. It contains run provenance, task metrics, TF-IDF logistic-regression baselines, segmented metrics, training history, uncertainty intervals, and merged review-level predictions.
- `article_analysis.csv`: normalized tabular analysis used by the manuscript. It is generated directly from `analysis.json` by the package during the same run.

Figures are not stored in the repository. The manuscript and dashboard should render charts from these canonical data artifacts when needed.

## Evaluation scope

Reported metrics include accuracy, precision, recall, macro and weighted F1, log-loss, Brier score, calibration error, segmented robustness metrics, a TF-IDF logistic-regression baseline, and 95% Wilson intervals for accuracy. The bundled corpus is synthetic. Results therefore measure behavior under the controlled generator distribution and are not evidence of external real-world generalization.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
lstm-pipeline train --run-id local --epochs 20
```

Python 3.12 · TensorFlow 2.20

## Citation

Citation metadata are provided in `CITATION.cff`. No permissive software license has been declared; reuse beyond applicable default copyright permissions requires authorization from the copyright holder.
