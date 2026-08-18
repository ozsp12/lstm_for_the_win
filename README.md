# LSTM for the Win

Reproducible continual-learning experiment for Long Short-Term Memory (LSTM) classification of synthetic product reviews by sentiment and product topic.

**Project page:** https://ozsp12.github.io/en/projects/lstm_ftw/

## Data lifecycle

`data/input/train.csv` is the cumulative training corpus. `data/input/incoming.csv` contains the current unseen synthetic batch across five linguistic levels, with variable length, slang, spelling noise, profanity, emojis, and mixed sentiment. After an evaluation, only rows marked `goldtest=1` are promoted to the training corpus; the next incoming batch is then generated. Existing training rows are not automatically deleted when the text-generator version changes.

`data/input/benchmark.csv` is created once from non-gold rows of the first incoming batch observed after benchmark support is introduced. Because those rows have `goldtest=0`, they are never promoted into training. The file is then treated as immutable and is used as a longitudinal synthetic benchmark across later runs.

Workflow runs are incremental. Each run receives its own timestamped directory under `data/output/`, and previous run directories are retained. `data/output/latest.json` is only a pointer to the newest completed run.

## Run artifact

Each new run persists exactly one analytical artifact:

- `run.json`: immutable canonical record of the experiment. It contains provenance, input hashes, package versions, parameters, validation-split metadata, primary LSTM and TF-IDF logistic-regression metrics, segmented metrics, training history, confidence intervals, merged review-level predictions, multi-seed summaries, and the immutable-benchmark evaluation.

No figures, models, duplicate metrics files, prediction CSVs, manifests, or paper-specific analytical tables are stored per run. Any later CSV or Parquet representation is a derived view that can be reconstructed from `run.json` when needed.

The intended data flow is therefore:

```text
train.csv + incoming.csv + benchmark.csv
                 |
                 v
              pipeline
                 |
                 v
 data/output/<run_id>/run.json
          /                 \
     dashboard          paper analysis
```

## Experimental safeguards

The internal validation split prefers holding out an entire coarse sentence-template family instead of randomly mixing structurally similar generated phrases between fit and validation. If a dataset does not contain enough distinguishable template families, the software falls back to deterministic label-stratified random splitting and records that fallback in `run.json`.

Production runs use a fixed split seed and multiple model seeds. This separates the structural validation partition from model-initialization randomness and allows run-level reporting of mean, population standard deviation, minimum, and maximum metrics across replicates. Accuracy also includes a 95% Wilson confidence interval for the evaluated sample.

## Evaluation scope

Reported metrics include accuracy, precision, recall, macro and weighted F1, log-loss, Brier score, calibration error, segmented robustness metrics, and a TF-IDF logistic-regression baseline. The bundled corpus and immutable benchmark are synthetic. Results therefore measure behavior under the controlled generator distribution and are not evidence of external real-world generalization.

A real external benchmark should be added only as a separately sourced, licensed, immutable evaluation dataset; it must never enter the synthetic promotion loop.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
lstm-pipeline train --run-id local --epochs 20 --replicate-seeds "42,1337,2026"
```

Python 3.12 · TensorFlow 2.20

## Citation

Citation metadata are provided in `CITATION.cff`. No permissive software license has been declared; reuse beyond applicable default copyright permissions requires authorization from the copyright holder.
