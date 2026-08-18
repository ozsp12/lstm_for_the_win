# LSTM for the Win

Reproducible continual-learning experiment for Long Short-Term Memory (LSTM) classification of product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Architecture

```text
agents/synthetic_data.py  -> cumulative train.csv and refreshed incoming.csv
handler.py                -> controlled state transitions
template_metadata.py      -> persisted generator-family metadata
classification/           -> LSTM, baseline, metrics and structural validation split
benchmark.py              -> immutable synthetic longitudinal benchmark
external_benchmark.py     -> immutable real-world UCI sentiment benchmark
experiment.py             -> experiment orchestration
run_artifact.py           -> canonical run.json
derived_artifacts.py      -> article_analysis.csv and figures/ from run.json only
cli.py                    -> command-line parsing
```

## Data lifecycle

`data/input/train.csv` is cumulative. After every successful evaluation, rows marked `goldtest=1` are promoted and the file is overwritten at the same path with a larger row count. `data/input/incoming.csv` is overwritten with the next unseen synthetic batch. Each generated record carries a persisted `template_family`, which is used for family-level validation splitting.

`data/input/benchmark.csv` is an immutable synthetic longitudinal benchmark built from non-gold rows and never promoted into training. `data/external/` is evaluation-only and contains the Amazon subset of the UCI **Sentiment Labelled Sentences** dataset (DOI `10.24432/C57604`, CC BY 4.0). External validation is sentiment-only because the source has no compatible topic labels and no neutral class.

## Output contract

`run.json` is the canonical source of truth. `article_analysis.csv` and every file in `figures/` are deterministic derived artifacts generated exclusively from the same `run.json`.

```text
data/output/
├── latest.json
└── <timestamp>_github-<run_id>/
    ├── run.json
    ├── article_analysis.csv
    └── figures/
```

Only the latest fully validated run is retained. The previous run remains present while the new run is being trained and checked. It is removed only after the new run bundle passes contract validation, deterministic regeneration, automated tests and the coverage gate. `latest.json` is then committed together with the new run.

`article_analysis.csv` is intended for human inspection and tabular analysis. It contains aggregate and segment metrics, uncertainty, confusion matrices, training history, benchmark/external results and one `prediction_record` row for every incoming observation. It never performs an independent calculation: all values come from `run.json`.

The figure directory is overwritten during derivation and rebuilt from the new `run.json`. Deleting `article_analysis.csv` and `figures/` and regenerating them must reproduce identical SHA-256 hashes in the same locked environment; CI enforces this invariant.

## Experimental safeguards

The validation split prefers holding out an entire persisted sentence-template family. Production uses a fixed split seed plus model seeds `42`, `1337` and `2026`. Runs report Wilson intervals for sample accuracy, Student-t intervals across model seeds, and an exact two-sided McNemar comparison between the LSTM and TF-IDF logistic-regression baseline.

The synthetic benchmark is immutable and separate from continual learning. The UCI Amazon subset adds independent real-world evidence for sentiment only; topic remains synthetic-only.

## Reproducible environment

`requirements-lock.txt` is fully resolved and hash-locked. CI and production install with `--require-hashes`, use the pinned build backend without build isolation, run `pip check`, and pin GitHub Actions to immutable commit SHAs. Test coverage must remain at least 90%.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
lstm-pipeline train --run-id local --epochs 20 --replicate-seeds "42,1337,2026"
python -m lstm_for_the_win.derived_artifacts data/output/local/run.json
```

Python 3.12 · TensorFlow 2.20

## Citation

Citation metadata are provided in `CITATION.cff`. No permissive software license has been declared; reuse beyond applicable default copyright permissions requires authorization from the copyright holder.
