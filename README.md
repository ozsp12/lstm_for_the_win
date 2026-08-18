# LSTM for the Win

Reproducible continual-learning experiment for Long Short-Term Memory (LSTM) classification of product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Architecture

```text
agents/improved_synthetic_data.py -> cumulative train.csv and refreshed incoming.csv
handler.py                         -> controlled input-state transitions
classification/                    -> LSTM, baseline, metrics and structural validation
benchmark.py                       -> immutable synthetic longitudinal benchmark
external_benchmark.py              -> immutable real-world UCI sentiment benchmark
experiment.py                      -> frozen-state experiment orchestration
run_artifact.py                    -> canonical run.json
derived_artifacts.py               -> wide article_analysis.csv and figures/ from run.json only
cli.py                             -> command-line interface
```

## Data lifecycle

`data/input/train.csv` is cumulative. `data/input/incoming.csv` represents the current unseen synthetic batch. New records receive `template_family` directly at render time; the value is then persisted and used for family-level validation splitting. The generator also varies stratum sizes, mixed-sentiment placement, linguistic structure and spelling noise so the synthetic corpus is not perfectly regular.

Training and data advancement are separate operations. The production experiment always trains on a frozen committed input state. `.github/workflows/advance-data.yml` performs the optional next-generation transition: approved `goldtest=1` rows are promoted into `train.csv`, a fresh `incoming.csv` is generated, and that committed state then triggers a new frozen-state experiment. A reset publishes generation 0 without immediately promoting any rows.

`data/input/benchmark.csv` is an immutable synthetic benchmark built from non-gold incoming rows and never promoted into training. `data/external/` contains the Amazon subset of UCI **Sentiment Labelled Sentences** (DOI `10.24432/C57604`, CC BY 4.0). External validation is sentiment-only because the source has no compatible topic labels and no neutral class.

## Output contract

`run.json` is the canonical source of truth. `article_analysis.csv` and every file in `figures/` are deterministic derivatives generated exclusively from the same run.

```text
data/output/
├── latest.json
└── <timestamp>_github-<run_id>/
    ├── run.json
    ├── article_analysis.csv
    └── figures/
```

Only the latest fully validated run is retained. The previous run is removed only after the replacement passes contract validation, deterministic regeneration, automated tests and the coverage gate.

`article_analysis.csv` is a conventional wide table for human inspection: one row per incoming observation, with review metadata, predictions, canonical model metrics, baseline metrics, paired-test information, benchmark accuracy and external-validation summaries in columns. It performs no independent calculation.

## Experimental safeguards

The validation split prefers holding out an entire persisted sentence-template family. Production uses model seeds `42`, `1337` and `2026`; means across those seeds are the canonical comparable metrics, with Student-t 95% intervals. Primary-seed results remain in the artifact for individual predictions, training history and confusion matrices.

Each task records the confusion-matrix convention explicitly as rows = expected and columns = predicted. Segment accuracy includes Wilson 95% intervals and support. Expected calibration error (ECE) is accompanied by the ten reliability bins used to compute it. LSTM and TF-IDF logistic regression are compared with an exact two-sided McNemar test.

The external Amazon evaluation reports both the native three-class model result and a source-compatible binary result obtained by restricting and renormalizing probabilities to `{negative, positive}`. Full probability vectors are preserved for external observations. This distinction prevents neutral predictions from being conflated with performance under the binary source label space.

## Reproducibility

The environment is hash-locked, SciPy is a direct dependency, GitHub Actions are pinned by immutable SHA, and test coverage must remain at least 90%. TensorFlow deterministic operations are enabled and `PYTHONHASHSEED`, deterministic-operation state, model seeds and split seed are recorded or enforced by CI. Metric implementations are regression-tested against `sklearn.metrics` reference implementations.

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

## Licensing and citation

Repository software is licensed under the MIT License. External dataset licensing and attribution are documented separately in `DATA_LICENSES.md`; the UCI Amazon subset remains CC BY 4.0. Citation metadata are provided in `CITATION.cff`, and CI verifies that citation, package, project and pipeline versions remain synchronized.
