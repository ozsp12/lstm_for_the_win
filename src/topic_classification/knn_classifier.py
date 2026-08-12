#!/usr/bin/env python
# coding: utf-8

# In[3]:


###################################################################################################################
# Requirements

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
import emoji
import spacy
import string
import unicodedata
import datetime
import random
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

###################################################################################################################
# chama o dataframe original

# transformando e abrindo o arquivo em formato csv
df = pd.read_excel('DateToTestSentiment_20210817.xls.xls')

###################################################################################################################
# Funções

def preprocess_data(data, 
                    columns,
                    null = True):
    
    df = data[columns]
    
    if null:
        df = df.dropna().reset_index().drop(columns=['index'])
    
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
    
    # corrects the bug that eliminates letters from words with accents
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text) 
    if not unicodedata.combining(ch))

    # removes special characters 
    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        text = re.sub(r"#(\w+)", " ", text)

    #
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    words = (''.join(nopunct)).split()
    
    # removes stopwords with less than two letters
    if(remove_stop):
        words = [w for w in words if w not in portuguese_stopwords]
        words = [w for w in words if len(w) > 2]  

    if(stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words)


def tokenizacao(df):
    
    # gets the number of rows and columns from the dataframe
    rows, cols = df.shape

    # creates the dataset column with the vectorized text
    df['token'] = [preprocess_text(df["COMMENT_TEXT"][row]) for row in range(rows)]
    
    return df

portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')

# escolhe as colunas que usaremos e remove as linhas nulas
df2 = preprocess_data(df, 
                      columns=['KEY','COMMENT_ID','COMMENT_TEXT'],
                      null = True
                     )

# cria coluna com comentários tokenizados
df2 = tokenizacao(df2)

# creates an intermediary dataframe
df3 = df2.copy()

# remove as linhas que a tokenização ficou com tamanho 0 (sem texto)
for n, item in enumerate(df2.token):
    if len(item)==0:
        df3 = df3.drop(n)
        
        
topics = ['AQUECIMENTO', 
          'ASSISTÊNCIA TÉCNICA', 
          'ATENDIMENTO', 
          'AUTO FALANTE', 
          'BATERIA', 
          'CAMERA', 
          'CARREGADOR', 
          'CUSTO BENEFICIO', 
          'DESIGN', 
          'ENTREGA', 
          'FLASH', 
          'FONE', 
          'JOGOS', 
          'MEMÓRIA',
          'PESO', 
          'PREÇO', 
          'PROCESSADOR', 
          'QUALIDADE', 
          'RESISTÊNCIA', 
          'TAMANHO', 
          'TELA', 
          'TRAVAMENTO',
          'VELOCIDADE', 
          'GENÉRICO/OUTRO'
         ]

# cria dicionário associndo um número de 0 a 23 a cada tópico na ordem da lista
dic_topics = {}
for n, item in enumerate(topics):
    dic_topics[n]=item

# tokeniza os tópicos da lista
topics_token = [preprocess_text(item) for item in topics]


# defines empty list
lista_clusters = []

# cria clusters onde cada um deles possui comentários contendo um dos tópicos da lista, caso um comentário contenha dois
# dos tópicos, a ordem de prioridade é a mesma ordem na qual os tópicos estão na lista.
for n, objeto in enumerate(topics_token):
    
    if len(objeto) == 1:
        lista_1 = [item for item in df3.token if objeto[0] in item]
        for n, item in enumerate(lista_clusters):
            lista_1 = [item1 for item1 in lista_1 if item1 not in item]
        lista_clusters.append(lista_1)

    else:
        lista_2 = [item for item in df3.token if (
            objeto[0] in item) and (objeto[1] in item)]
        for n, item in enumerate(lista_clusters):
            lista_2 = [item1 for item1 in lista_2 if item1 not in item]
        lista_clusters.append(lista_2)
        

###################################################################################################################

temp = []

# junta todos os comentários de todos os tópicos em uma única lista sem divisão
for n, objeto in enumerate(lista_clusters):
    temp.extend(objeto)

# cria um último clusters formado por todos os comentários que não possui os elementos da lista acima, isto é não possui
# nenhum dos tópicos
ultimo_cluster = [item for item in df3.token if item not in temp]

# define esse último cluster como último elemento da lista de clusters, que vai servir como o tópico "genérico/outro"
lista_clusters[23] = ultimo_cluster

# cria uma copia da lista_clusters
lista_clusters1 = lista_clusters.copy()

# toma um sample de 2000 comentários para os clusters que tiveram mais que 3500 comentários, diminuindo o balanceamento
for n, item in enumerate(lista_clusters1):
    if len(item) > 3500:
        lista_clusters1[n] = random.sample(item, 2000)    
        
        
# carrega as arquivo "pt_core_nes_md" do spacy, usado para vetorização de textos em português
nlp = spacy.load('pt_core_news_md')

# define a vetorização usando a função str.vector do spacy
def vec(s):
    return nlp.vocab[s].vector


# cria uma lista com a quantidade de elementos em cada um dos clusters
linhas = [len(item) for item in lista_clusters1]

# tamanho das palavras vetorizadas
vec_size = 300

# soma a quantidade de elementos de todos os clusters
rows = sum(linhas)

# cria uma lista vazia
list_of_matrix = [] 

# cria uma matriz vazia para ser preenchida com os comentários vetorizados ao final
final_feature_matrix = np.empty([rows, vec_size])

# cria preenche a matriz vazia criada acima com os comentários vetorizados
for n, item in enumerate(lista_clusters1):
    for corpus in item: 
        matrix = np.empty([len(corpus), vec_size]) 
                                              
        for idx, word in enumerate(corpus):
            matrix[idx,:] = vec(word) 
        list_of_matrix.append(matrix)

# cama comentário vetorizado possui 300 variáveis tiramos a média de cada palavra em cada frase para preencher a matriz
# criada acima e preenche-la com esses vetores
for row in range(rows):
    final_feature_matrix[row,:] = list_of_matrix[row].mean(axis = 0)
    

labels = []

# criamos uma lista com os labels de 0 a 23 para cada tópico
for n, item in enumerate(lista_clusters1):
    for i, objeto in enumerate(item):
        labels.append(n)
    
# transformamos em um array
x = np.array(labels)

# fazemos o reshape para concatenar a matriz de variáveis
x = x.reshape(-1,1)

#  concatenamos a matriz de variáveis com os labeis
final_matrix = np.concatenate((final_feature_matrix, x), axis=1)

# fazemos um shuffle para os comentários não ficar na sequencia pelos labels
final_matrix1 = shuffle(final_matrix, random_state = 42)

# dividindo em treino e teste
Xtreino, Xteste, ytreino, yteste = train_test_split(
    final_matrix1[:, 0:-1], final_matrix1[:, -1], train_size = 0.7, random_state=42)

# definindo o knn com 2 vizinhos
knn = KNeighborsClassifier(n_neighbors = 2)

# treinando o knn
knn.fit(Xtreino, ytreino)

# fazendo a predição
pred = knn.predict(Xteste)

# vendo a acurácia
accuracy_score(yteste,pred)


###################################################################################################################
# Cria dataframe

def classifica_dataframe(df):
    
    df2 = preprocess_data(df, 
                      columns=['KEY','COMMENT_ID','COMMENT_TEXT'],
                      null = True
                     )
    
    df2 = tokenizacao(df2)
    
    for n, item in enumerate(df2.token):
        if len(item)==0:
            df2 = df2.drop(n)
    
    df2 = df2.reset_index().drop(columns={'index'})
    lista_clusters = []
    
    for n, objeto in enumerate(topics_token):
        
        if len(objeto) == 1:
            lista_1 = []
            for i in range(len(df2)):
                if (objeto[0] in df2.token[i]):
                    lista_1.append([df2.token[i], df2.KEY[i]])
            for n, item in enumerate(lista_clusters):
                lista_1 = [item1 for item1 in lista_1 if item1 not in item]
            lista_clusters.append(lista_1)

        else:
            lista_2 = []
            for i in range(len(df2)):
                if (objeto[0] in df2.token[i]) and (objeto[1] in df2.token[i]):
                    lista_2.append([df2.token[i], df2.KEY[i]])
            for n, item in enumerate(lista_clusters):
                lista_2 = [item1 for item1 in lista_2 if item1 not in item]
            lista_clusters.append(lista_2)
            
    temp = []
    for n, objeto in enumerate(lista_clusters[:23]):
        lista_temp = [item[0] for item in objeto] 
        temp.extend(lista_temp)
    ultimo_cluster = []
    for i in range(len(df2)):
        if (df2.token[i] not in temp):
            ultimo_cluster.append([df2.token[i], df2.KEY[i]])

    lista_clusters[23] = ultimo_cluster
    
    labels = []
    
    for n, item in enumerate(lista_clusters):
        for i, objeto in enumerate(item):
            labels.append(n)
    
    categorias = [dic_topics[item] for item in labels]
    
    dff = pd.DataFrame(lista_clusters[0])
    for n, objeto in enumerate(lista_clusters[1:]):
        dff = pd.concat([dff,pd.DataFrame(objeto)])
    
    dff = dff.rename(columns={1:'KEY'})
    
    dff['categoria'] = categorias
    
    df_final = pd.merge(df2,dff, on='KEY')
    df_final = df_final[['KEY','COMMENT_ID','COMMENT_TEXT','categoria']]
    
    return df_final

# chama a função que cria o dataframe final
df3 = classifica_dataframe(df)

