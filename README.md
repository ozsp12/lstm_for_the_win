# LSTM for the Win

Reproducible continual-learning experiment for Long Short-Term Memory (LSTM) classification of product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Architecture

The package separates persistent state, experiment execution, model code, and artifact construction:

```text
agents/synthetic_data.py  -> synthetic corpus state
handler.py                -> controlled state transitions
template_metadata.py      -> persisted generator-family metadata
classification/           -> LSTM, baseline, metrics and validation split
benchmark.py              -> immutable synthetic longitudinal benchmark
external_benchmark.py     -> immutable real-world UCI sentiment benchmark
experiment.py             -> experiment orchestration
run_artifact.py           -> canonical run.json construction
cli.py                    -> command-line parsing
```

`handler.py` remains the stable console boundary and delegates scientific execution to `experiment.py`.

## Data lifecycle

`data/input/train.csv` is the cumulative synthetic training corpus. `data/input/incoming.csv` contains the current unseen synthetic batch across five linguistic levels, with variable length, slang, spelling noise, profanity, emojis, and mixed sentiment. After evaluation, only rows marked `goldtest=1` are promoted to training; the next incoming batch is then generated. Existing training rows are not automatically deleted when the text-generator version changes.

Each generated record has a persisted `template_family`. Legacy rows are backfilled once and the value is then stored in the CSV. Validation therefore groups by metadata rather than inferring structural families during the split.

`data/input/benchmark.csv` is an immutable synthetic longitudinal benchmark created from non-gold incoming rows. `benchmark_manifest.json` records its bootstrap generation, provenance and SHA-256. Benchmark rows never enter training.

`data/external/` is strictly evaluation-only. The pipeline uses the Amazon subset of the UCI **Sentiment Labelled Sentences** dataset (DOI `10.24432/C57604`, CC BY 4.0) as an immutable real-world sentiment benchmark. The source contains binary positive/negative labels and no compatible four-class product-topic labels, so external validation is reported for sentiment only. External data never enter the continual-learning loop.

Workflow runs are incremental. Each run receives its own timestamped directory under `data/output/`, previous runs are retained, and `data/output/latest.json` only points to the newest completed run.

## Run artifact

Each new run persists exactly one analytical artifact:

- `run.json`: immutable canonical record containing provenance, input hashes, environment versions, parameters, structural split metadata, LSTM and TF-IDF logistic-regression metrics, exact paired McNemar comparison, segmented metrics, training history, Wilson intervals, multi-seed summaries with between-seed intervals, review-level predictions, synthetic benchmark evaluation, and real external sentiment evaluation.

No figures, models, duplicate metrics files, prediction CSVs, manifests, or paper-specific analytical tables are stored per run. CSV or Parquet representations are derived views reconstructed from `run.json` when needed.

```text
synthetic train/incoming + immutable synthetic benchmark + external sentiment benchmark
                                   |
                                   v
                               pipeline
                                   |
                                   v
                     data/output/<run_id>/run.json
                            /                    \
                       dashboard             paper analysis
```

## Experimental safeguards

The internal validation split prefers holding out an entire persisted sentence-template family instead of mixing structurally similar generated phrases between fit and validation. If a corpus does not support a valid family-level holdout, the fallback is deterministic label-stratified random splitting and the fallback is recorded in `run.json`.

Production runs use a fixed split seed and multiple model seeds. The run reports mean, population and sample standard deviations, range, and a 95% Student-t interval across model seeds for comparable metrics. Accuracy also includes a 95% Wilson interval for each evaluated sample. LSTM and TF-IDF baseline correctness are compared on the same incoming observations with an exact two-sided McNemar test.

## Reproducible environment

`pyproject.toml` is the high-level package specification. `requirements-lock.in` defines the direct reproducible environment and `requirements-lock.txt` is the fully resolved hash-locked installation set. Validation and production install only hash-verified dependencies, use the pinned build backend without build isolation, and run `pip check` before model execution. GitHub Actions are pinned to immutable commit SHAs.

## Evaluation scope

Synthetic incoming and synthetic benchmark results characterize the controlled generator distribution. The UCI Amazon subset adds independent real-world evidence for **sentiment only**. Because that source has no compatible topic labels and no neutral class, it does not establish external topic generalization or full three-class sentiment coverage. These limitations are recorded in every new `run.json`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
lstm-pipeline train --run-id local --epochs 20 --replicate-seeds "42,1337,2026"
```

Python 3.12 · TensorFlow 2.20

## Citation

Citation metadata are provided in `CITATION.cff`. No permissive software license has been declared; reuse beyond applicable default copyright permissions requires authorization from the copyright holder.
