# External evaluation data

This directory is reserved for immutable, independently sourced evaluation data. External data must never enter the synthetic continual-learning promotion loop.

The production pipeline bootstraps the Amazon subset of the UCI **Sentiment Labelled Sentences** dataset (dataset DOI `10.24432/C57604`) into `data/external/uci_sentiment_labelled_sentences/` when it is absent. UCI distributes the dataset under CC BY 4.0. The external benchmark is used only for sentiment evaluation because it contains binary positive/negative labels and no compatible product-topic labels.
