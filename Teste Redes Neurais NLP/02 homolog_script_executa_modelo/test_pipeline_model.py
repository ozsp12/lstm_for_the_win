import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
import emoji
import spacy
import string
import tensorflow as tf
import unicodedata
import datetime
from sklearn.utils import shuffle
from joblib import dump, load
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from joblib import load


def sentimento(x):
    """
    Função que transforma a coluna passada como argumento
    na variável target.
    """
    
    if x == 5:
        return 2
    elif x == 4:
        return 0
    else:
        return 1
    
    

def balance_classes(df):
    
    n4 = len(df[df['RATING_VALUE'] == 4])
    
    #
    df1 = df.loc[df['RATING_VALUE'] == 5, :]

    #
    df2 = df.loc[df['RATING_VALUE'] != 5, :]

    #
    df1 = df1.sample(n4)

    # contrói a tabela final
    df_final = df1.append(df2).reset_index().drop(columns=['index'])
    
    return df_final

def preprocess_data(data, columns,
                    label_column, 
                    null=True,
                    labels_ok=True, 
                    dict_labels=None,
                    labels_ready=True, 
                    labels_true=None):
    """ Função para tratamento de datasets para analise de sentimentos
    
    Args:
        data (pandas dataset) = Dataset 
        columns (list) = nome da coluna com texto para analise e da coluna com os labels
        null (Boolean) = define se vai limpar ou não os dados faltantes.
        label_columns (str)= nome da coluna com os label
        label_ok (Boolean) = define se a coluna de label está tratada ou não
        dict_label (dict) = define pelas chaves e valores as substituições vão ser feitas na coluna de labels.
        labels_ready (Boolean) = define se vamos usar os labels sem nenhuma alteração
        labels_true (dict) = dicionario com chaves sendo tuplas (com pelo menos dois elementos,
        mesmo que sejam elementos repetidos), com labels e valores sendo os respectivos 
        sentimentos (na ordem: positive, neutral e negative).
    """
    df = data[columns]
    
    if null:
        df = df.dropna().reset_index().drop(columns=['index'])
    
    if not labels_ok:
        for n, item in enumerate(df[label_column]):
            if type(item) == str:
                for key, value in dict_labels.items():
                    df[label_column][n] = re.sub(str(list(key)),value, df[label_column].iloc[n])
        df[label_column] = df[label_column].astype('float32')
                    
    if not labels_ready:
        def sentimento(x):
            if (x in [*labels_true.keys()][0]):
                return [*labels_true.values()][0]
            elif (x in [*labels_true.keys()][1]):
                return [*labels_true.values()][1]
            else:
                return [*labels_true.values()][2]
            
    df['sentimentos'] = df[label_column].apply(sentimento)    
    
    return df


def preprocess_text(text, 
                    remove_stop = True, 
                    stem_words = False, 
                    remove_mentions_hashtags = True
                   ):
    """
    eg:
    input: preprocess_text("@water #dream hi hello where are you going be there tomorrow happening happen happens",  
    stem_words = True) 
    output: ['tomorrow', 'happen', 'go', 'hello']
    """

    # Remove emojis
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])
    
    # Corrige o bug que elimina parte das palavras com acentos
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text) 
    if not unicodedata.combining(ch))

    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        text = re.sub(r"#(\w+)", " ", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    words = (''.join(nopunct)).split()

    if(remove_stop):
        words = [w for w in words if w not in portuguese_stopwords]
        words = [w for w in words if len(w) > 2]  

    if(stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words)


def tokenizacao(df):
    
    # cria a coluna com textos vetorizados
    rows, cols = df.shape

    df['token'] = [preprocess_text(df["COMMENT_TEXT"][row]) for row in range(rows)]
    
    return df


def treinar_vetorizacao(df):

    # coleciona as palavras usadas para o treinamento da função CountVectorizer
    lista_treino = []

    for item in df['token']:
        lista_treino1 = [n for n in item if n not in lista_treino]
        lista_treino.extend(lista_treino1)

    # treina o modelo de vetorização
    vectorize = CountVectorizer(lowercase=True, strip_accents = 'unicode')

    vectorize.fit(lista_treino)
    
    return vectorize, lista_treino


def vectorize2(lista):
    lista2 = [vectorize.vocabulary_[item] for item in lista]
    
    return lista2


def separa_df_test_train(df, size):

    #
    df = shuffle(df, random_state=42)

    # separando as variaveis
    X = df['vectors'].values

    #
    y = df['sentimentos'].values

    # usa a função pad_sequences do keras para deixar todos os textos do mesmo tamanho
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=20)

    # separa os dados em treino e teste e fazendo o one-hot-encoding na variável y
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = size, random_state=42)

    #
    ytrain = tf.keras.utils.to_categorical(ytrain)

    #
    ytest = tf.keras.utils.to_categorical(ytest)

    return X, y, Xtrain, Xtest, ytrain, ytest


def redes_neurais_modelo():

    #
    model = tf.keras.Sequential()

    #
    model.add(tf.keras.layers.Embedding(input_dim = len(lista_treino) + 1, 
                                        output_dim = 128, 
                                        input_shape = (Xtrain.shape[1],), 
                                        activity_regularizer = regularizers.l2(1e-5)
                                       )
             )

    #
    model.add(tf.keras.layers.Dropout(0.25))

    #
    model.add(tf.keras.layers.LSTM(units = 512, activation='relu'))

    #
    model.add(tf.keras.layers.Dense(units = 256, activation='relu'))

    #
    model.add(tf.keras.layers.Dropout(0.25))

    #
    model.add(tf.keras.layers.Dense(units = 128, activation='relu'))

    #
    model.add(tf.keras.layers.Dropout(0.25))

    #
    model.add(tf.keras.layers.Dense(units = 64, activation='relu'))

    #
    model.add(tf.keras.layers.Dropout(0.25))

    #
    model.add(tf.keras.layers.Dense(units = 3, 
                                    activation='softmax', 
                                    activity_regularizer = regularizers.l2(1e-5)
                                   )
             )

    #
    model.compile(optimizer = 'Adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy']
                 )

    #
    model.summary()

    return model


def cria_stopper_rede(epo):

    # criando um stopper para a rede
    stopper = EarlyStopping(monitor="val_accuracy",
                             patience=5, 
                            verbose=2, 
                            mode='max'
                           )

    callbacks = [stopper]

    #
    history = model.fit(x = Xtrain, 
                        y = ytrain, 
                        batch_size = 128,
                        validation_data = (Xtest, ytest),
                        epochs = epo,
                        callbacks = callbacks
                        )
    
    return callbacks, history


def feelings_prediction(model):
    
    # organiza os labels pelo valor da rating dado pelo cliente
    dict_feeling = {2: 'Positive', 1: 'Negative', 0: 'Neutral'}

    text = input("Digite seu texto aqui\n\n")
    
    text1 = preprocess_text(text)
    
    text1 = [vectorize2(text1)]
    
    text1 = tf.keras.preprocessing.sequence.pad_sequences(text1, maxlen=20)
    
    pred = model.predict(text1)
    
    predd = dict_feeling[np.argmax(pred)]
    
    print('\nSentimento analisado:', predd)
    
    
def feelings_prediction_v02(model,df):
    
    # organiza os labels pelo valor da rating dado pelo cliente
    dict_feeling = {2: 'Positive', 1: 'Negative', 0: 'Neutral'}
       
    text = df['vectors'].values
    
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=20)
    
    lista_pred = model.predict(text)
    
    df['pred_sentiment'] = [dict_feeling[np.argmax(item)] for item in lista_pred]
    
    return df
	
	
	
portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')

vectorize = load('vetorize_sklearn.joblib')

model = load_model('primeiro_modelo.h5')

df = pd.read_excel('DateToTestSentiment_20210817.xls.xls')

df2 = df.sample(100)

#
df2 = preprocess_data(df2, 
                           columns=['KEY','COMMENT_ID','COMMENT_TEXT', 'RATING_VALUE'],
                           label_column='RATING_VALUE', 
                           null = True,
                           labels_ok = False,
                           dict_labels = {(' '): '', (','): '.'},
                           labels_ready = False, 
                           labels_true = {(5,5): 2, (4,4):0,(1,2,3):1}
                          )

# contruindo a coluna nova com os sentimentos
df2['sentimentos'] = df2['RATING_VALUE'].apply(sentimento)

#
df2 = tokenizacao(df2)

# criando a coluna com o texto vetorizado
df2['vectors'] = df2['token'].apply(vectorize2)

df2 = feelings_prediction_v02(model,df2)

df2.head()