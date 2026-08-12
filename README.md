# LSTM for the Win

Protótipos de NLP para classificar avaliações de produtos em português e espanhol.

O repositório reúne dois fluxos relacionados:

| Tarefa | Abordagem principal |
|---|---|
| Análise de sentimento | `Embedding -> LSTM(512) -> Dense -> Softmax`, com três classes |
| Classificação de tópicos | rotulagem assistida por regras, vetores do spaCy e KNN; K-Means para exploração |

> Status: projeto de pesquisa/legado. Os notebooks e seus resultados foram preservados, mas não existe ainda uma CLI única nem um ambiente integralmente validado.

## Pipeline

```text
avaliações -> limpeza/tokenização -> vetorização -> treino/teste -> classificação -> artefatos
```

O pré-processamento remove pontuação, emojis, menções, hashtags e stopwords. Os experimentos de sentimento usam sequências com comprimento 20 e saída `softmax` para positivo, neutro e negativo. Os classificadores de tópicos usam embeddings `pt_core_news_md` e KNN com `k=2`.

## Estrutura

```text
src/          código reutilizável
notebooks/    experimentos principais
data/         dados brutos, externos e processados
models/       modelos e vetorizadores treinados
references/   artigos e notas técnicas
docs/assets/  imagens da documentação
archive/      protótipos e versões históricas
```

## Ambiente

Recomenda-se Python 3.8 por compatibilidade com a pilha original.

```bash
python -m venv .venv
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download pt_core_news_md
jupyter lab
```

Comece por:

- `notebooks/topic_classification/knn/pt/` para classificação de tópicos;
- `notebooks/sentiment_analysis/pt/` para análise de sentimento com LSTM;
- `src/sentiment_analysis/lstm_train_model.py` para a implementação reutilizável do modelo.

## Resultados registrados

Os notebooks salvos registram acurácia de teste entre **68,16% e 69,83%** nos quatro classificadores KNN em português. Nos protótipos LSTM, a acurácia de validação registrada varia de **51,78% a 61,81%**. Esses números são históricos e não foram reexecutados nesta reorganização.

## Limitações

Alguns notebooks mantêm caminhos relativos e APIs antigas de pandas, scikit-learn e emoji; pequenos ajustes podem ser necessários em ambientes atuais. As bases e os modelos são grandes e continuam versionados para preservar o trabalho existente; para novas versões, prefira Git LFS ou armazenamento externo. O repositório não declara licença.
