# LSTM for Sentiment and Topic Classification

A minimal, reproducible project with two product-review classifiers:

- sentiment: `positive`, `neutral`, or `negative`;
- topic: `smartphone`, `television`, `refrigerator`, or `washing_machine`.

## Solution design

```mermaid
flowchart LR
    A["Product reviews"] --> B["Cleaning and tokenization"]
    B --> C{"Objective"}

    C --> D["Sentiment analysis"]
    D --> E["Vectorization and padding"]
    E --> F["Embedding + LSTM"]
    F --> G["Positive, neutral, or negative"]

    C --> H["Topic classification"]
    H --> I["spaCy language vectors"]
    I --> J["KNN by product category"]
    J --> K["Predicted topic"]

    H -. exploration .-> L["K-Means"]
```

The diagram also records the original spaCy/KNN topic-design and K-Means exploration path. The automated notebook currently runs both supervised tasks through the shared LSTM implementation.

## How it works

1. Loads and validates the sentiment and topic datasets.
2. Normalizes text and creates stratified train/test splits.
3. Learns an independent vocabulary for each task with `TextVectorization`.
4. Trains an `Embedding -> LSTM -> Dense -> Softmax` network for each classifier.
5. Evaluates both classifiers and produces sample predictions.

All reusable logic lives in `.py` files; the notebook only configures, runs, and presents the result.

## Project structure

```text
.github/workflows/pipeline.yml     automated GitHub Actions execution
data/sentiment_samples.csv         synthetic sentiment data
data/topic_samples.csv             synthetic topic data
notebooks/text_classification_pipeline.ipynb
src/text_classifier/               shared LSTM implementation
src/sentiment_classifier/          sentiment entry point
src/topic_classifier/              topic entry point
```

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/text_classification_pipeline.ipynb
```

The executed notebook retains metrics and predictions in its output cells.

## GitHub Actions

The `pipeline.yml` workflow runs on every push, pull request, or manual dispatch. It:

- installs the environment;
- validates Python module syntax;
- executes the notebook from start to finish;
- uploads the executed notebook as a workflow artifact.

The included datasets are synthetic and intended for demonstration. For real use, replace them with labeled, PII-free data while preserving the `text` and `label` columns.
