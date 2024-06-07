#%%1
'''
Dissertação de Mestrado Vagner Vilela
Códificação do experimento em Python
UFJF - PGMC - 2024 
Prof. Dr. Leonardo Goliatt

Versionamento

*   v1:  14/05/2024: Vagner: criação da estrutura do notebook e implementação de GMDH e médias dos indicadores
*   v2:  15/05/2024: Vagner: medição dos quatro modelos GMDH
*   v3:  15/05/2024: Vagner: execução no Spyder
*   v4:  27/05/2024: Vagner: limpeza de código
*   v5:  28/05/2024: Vagner: Implementação do GridSearchCV para o RIA
*   v6:  29/05/2024: Vagner: Implementação do GridSearchCV para os outros modelos
*   v7:  31/05/2024: Vagner: Gravando melhores hiperparametros em json
*   v8:  31/05/2024: Vagner: Gravando melhores folds em json
*   v9:  05/06/2024: Vagner: Buscando descrição do hiperparametro criterion e não seu end. de memória
                             Inserção de tempo no RIA
                             Limpeza de código
*   v10  06/06/2024: Vagner: Executando polynomial linear_cov com cada um dos criterions
                             Inserindo pontos mandatórios de execução                             

*******************   Definição do Problema (P1) ************************************

Formato
Dados tratados resultantes de ETL 

Tipo de dado
CSV

Nomes das colunas

1. SN: Número de série
2. WHP: Pressão na cabeça do poço (Wellhead Pressure) 
3. WFR: Taxa de fluxo de água (Water Flow Rate)
4. OFR: Taxa de fluxo de óleo (Oil Flow Rate)
5. GFR: Taxa de fluxo de gás (Gas Flow Rate)
6. WPD: Produção diária de água (Water Production Daily)
7. API: Índice de gravidade específica do petróleo (Oil API (American Petroleum Institute) gravity)
8. ID: Diâmetro interno do tubo (Inner Diameter)
9. WBHT: Temperatura no fundo do poço (Bottomhole Temperature)
11.BHP: Pressão de fundo de poço em escoamento (Flowing Bottomhole Pressure)

'''
#%%2

""" ********************  @ Importação de bibliotecas (P2) run ***************************"""

import sys
print(sys.version)

#Importação de bibliotecas
import pylab as pl
import pandas as pd
import numpy as np
pl.style.use('ggplot')
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#pip install gmdh
import gmdh
from gmdh import Ria,Mia,Combi,Multi, split_data, Criterion, CriterionType, PolynomialType

import json
import os

import requests
from io import StringIO

import itertools

import time

import winsound

from datetime import datetime

#%%7

""" *******************  @ Aquisição e limpeza dos dados (P3) run ************************ """

# URL do arquivo CSV
url = "https://drive.google.com/uc?id=1IK9TQx-XFg2brTBiXTS8fbmaHHNY3Hxz"

# Baixar o conteúdo do arquivo CSV
r = requests.get(url)

# Ler o conteúdo do arquivo CSV
df = pd.read_csv(StringIO(r.text))

# Exibindo o DataFrame
print(df)

#%%19

""" ******************* Higienização dos dados (P5) ************************ """

#Não houve necessidade de limpeza e/ou transformação.
#Também não há linhas com valores nulos.


#%%20

#Verificando valores nulos
df.isnull().sum()


#%%
# Exibindo o DataFrame com as colunas renomeadas
print(df)

#%%

""" ******************* Análise Exploratória dos Dados (P6) ************************ """

# Estatísticas básicas
summary_statistics = df.describe()

# Exibindo as estatísticas
print("Estatísticas básicas para cada coluna:")
print(summary_statistics)

"""Indicativo de necessidade de escalonamento (feature scaling)"""

#%%

#Visualizando as colunas do dataset
df.info()

#%%

#Tabela de freqüência da variável alvo
df['BHP'].value_counts()

# Visualizar a distribuição da coluna alvo "BHP"
plt.figure(figsize=(10, 6))
sns.histplot(df['BHP'], kde=True, bins=30, color='blue')
plt.title('Distribuição de BHP')
plt.xlabel('BHP')
plt.show()

#%%

df.head()

#%%

"""###Análise de correlações

**Análise principal de correlação**</br>

Matriz de correlação entre todos os atributos numéricos, exceto o FBHP. Utilizando a biblioteca pandas para calcular a matriz de correlação e a seaborn para criar um mapa de calor para uma visualização mais clara.
"""

df_corr = df.drop(columns=['BHP','SN'])
df_corr.corr()

#%%

# Calculando a matriz de correlação
correlation_matrix = df_corr.corr()

# Criando um mapa de calor para visualização
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação entre Atributos')
plt.show()

#%%

# Calcula a matriz de correlação
correlation_matrix = df_corr.corr()

# Encontra pares de colunas com correlação maior do que 0.7
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
            highly_correlated_pairs.append(pair)

# Exibe os pares de colunas com correlação maior do que 0.7
if len(highly_correlated_pairs) > 0:
    print("Pares de colunas com correlação maior do que 0.7:")
    for pair in highly_correlated_pairs:
        print(pair)
else:
    print("Não há pares de colunas com correlação maior do que 0.7.")

"""Fluxo de gás e fluxo de óleo com alta correção"""

#%%

# Colunas numéricas para análise de outliers
numeric_columns = ['WHP', 'WFR', 'OFR', 'GFR', 'WPD', 'API', 'ID', 'WBHT', 'BHP']


# Identificação e visualização de outliers usando boxplot
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    df.boxplot(column=[column])
    plt.title(f'Boxplot para {column}')
    plt.show()

    # Calculando estatísticas resumidas para identificar potenciais outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f'\nAnálise de outliers para a coluna {column}:')
    print(f' - Número de outliers: {len(outliers)}')
    print(f' - Limite inferior: {lower_bound}')
    print(f' - Limite superior: {upper_bound}')
    print(f' - Valores outliers: {outliers[column].tolist()}')     

#%%

""" ******************* @ Escalonamento run ************************ """

# Exibindo as primeiras linhas do DataFrame antes da do escalonamento
print(df.head())

df_esc = df.copy()

# Selecionando apenas os atributos para a padronização (excluindo 'SN' e 'FBHP' e o id)
atributos_para_padronizar = df_esc.columns.difference(['SN', 'BHP'])

# Criando o objeto StandardScaler
scaler = StandardScaler()

# Aplicando a padronização nos atributos selecionados
df_esc[atributos_para_padronizar] = scaler.fit_transform(df_esc[atributos_para_padronizar])

# Exibindo as primeiras linhas do DataFrame após escalonamento
print(df_esc.head())

# Estatísticas básicas sem escalonamento
summary_statistics = df.describe()

# Exibindo as estatísticas
print("Estatísticas básicas para cada coluna:")
print(summary_statistics)

# Estatísticas básicas escalonada
summary_statistics = df_esc.describe()

# Exibindo as estatísticas
print("Estatísticas básicas para cada coluna:")
print(summary_statistics)

#%%

""" ******************* @ Separando features e variável de interesse (P24) run ************************ """

#Fazendo uma cópia de X para manter o id_medicao
X_com_id_medicao = df_esc.values

# Remover a coluna BHP que será minha variável de interesse
X = df_esc.drop(columns=['BHP','SN']).values
y = df_esc['BHP'].values


#%%
len(X)

#%%
len(y)
#%%
""" ******************* @ Separando treino e teste (P7) run ************************ """

seed = 42

""" GMDH """

# Dividindo a base de dados em treinamento (70%) e teste (30%)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2,random_state=seed)#ria 0,8503
#X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5,random_state=seed) #ria 0,82
#X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25,random_state=seed)#ria 0,8461
estratificacao = '0.2'

#%%
type(X)

#%%
print('Tamanho X treino:', len(X_train))
#%%
print('Tamanho y treino:',len(y_train))
#%%
print('Tamanho X teste:',len(X_test))
#%%
print('Tamanho y teste:',len(y_test))
#%%

# print result arrays
print('X_train:\n', X_train)
print('X_test:\n', X_test)
print('\ny_train:\n', y_train)
print('y_test:\n', y_test)

#%%

""" ******************* @ Funções de métricas de erro (P9) run ************************ """

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%%

""" ******************* @ Validação cruzada (P12) run ************************ """

# Definindo o número de folds
num_folds = 5 #5 = 0,81

# Inicializando o KFold
kf = KFold(n_splits=num_folds)


#%%
""" ****************     EXECUÇÃO DOS MODELOS (P14) ********************* """
    
#%%
len(y_test) #239
#%%
array_json = json.dumps(y_test.tolist())
#%%
type(X)
#%%
X[175,0].astype(int)

#%%

""" ****************     TESTE MANUAL COM ALGUNS HIPERPARAMETROS (P30) ********************* """

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução teste manual: ', time_formatted)

# Treinando o modelo
model = Ria()
#model.fit(X_train, y_train,k_best=0.01, p_average=1,n_jobs=1,verbose=0,test_size=-1,criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY))
#model.fit(X_train, y_train,k_best=0.01, p_average=1,n_jobs=-1,verbose=0,test_size=-1,criterion=CriterionType.STABILITY)#erro
#model.fit(X_train, y_train,k_best=25, p_average=15,n_jobs=-1,verbose=0,test_size=0.4,criterion=gmdh.CriterionType.REGULARITY)#ERRO
#model.fit(X_train, y_train,k_best=25, p_average=15,n_jobs=-1,verbose=0,test_size=0.4,criterion=REGULARITY)#ERRO
#model.fit(X_train, y_train,k_best=25, p_average=15,n_jobs=-1,verbose=0,test_size=0.4,criterion=gmdh.gmdh.CriterionType(0))#ERRO
#model.fit(X_train, y_train,k_best=0.01, p_average=1,n_jobs=1,verbose=0,test_size=-1,criterion=Criterion(CriterionType.STABILITY))
#model.fit(X_train, y_train,criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY),k_best=24, p_average=1,n_jobs=-1,verbose=0,test_size=0.01,polynomial_type=gmdh.PolynomialType.QUADRATIC)#Multi não tem polynomial_type
model.fit(X_train, y_train,criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY), k_best = 4, p_average=1,n_jobs=-1,verbose=0,test_size=0.25,polynomial_type=gmdh.PolynomialType.QUADRATIC)
#model.fit(np.array(X_train_fold), np.array(y_train_fold))
    
# Fazendo previsões
y_pred = model.predict(X_test)

"""alimenta tabela de execuções no bd""" 
# Converter o array para uma string JSON
#y_test_json = json.dumps(y_test_fold.tolist())
#y_pred_json = json.dumps(y_pred.tolist())                            

# Armazenando avaliações
#r2
score = r2_score(y_test, y_pred)
metrica_r2.append(score)    

#mse
score = mean_squared_error(y_test, y_pred)
rmse_calculado = np.sqrt(score)
metrica_mse.append(score)

#rmse
metrica_rmse.append(rmse_calculado)

#mape
score =  mape(y_test, y_pred)
metrica_mape.append(score) 

#id_resultado
#id_resultado.append(valor_id_resultado) 

# Guardando hiperparametros
#inserir_dados(df_hiper_ria, valor_id_resultado) 

# Colocando métricas em um DataFrame
df_metricas = pd.DataFrame({
    'R2': metrica_r2    
    ,'MSE': metrica_mse
    ,'RMSE': metrica_rmse
    ,'MAPE': metrica_mape
    #,'id_resultado': id_resultado
})


print("Métricas:")
print(df_metricas)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

print(f"Tempo de Execucao: {execution_time} segundos")

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%
model.get_best_polynomial()

#%%
# Função para serializar os hiperparâmetros - (P31) - run
def serialize_hyperparameters(criterion, p_average, n_jobs, test_size, k_best=None, polynomial_type=None):
#def serialize_hyperparameters(criterion, p_average, n_jobs, test_size, k_best=None, polynomial_type=None):
    serialized = {
        #'criterion': str(criterion),
        'criterion': criterion.criterion_type.name,
        #'criterion': criterion.criterion_type.name if isinstance(criterion, Criterion) else str(criterion),
        'p_average': p_average,
        'n_jobs': n_jobs,
        'test_size': test_size
    }
    if k_best is not None:
        serialized['k_best'] = k_best
    if polynomial_type is not None:
        serialized['polynomial_type'] = str(polynomial_type)
    return serialized

#%%
print(gmdh.Criterion(gmdh.CriterionType.STABILITY))
#%%
print(gmdh.Criterion(gmdh.CriterionType.REGULARITY))
#%%
print(gmdh.CriterionType.REGULARITY)
#%%
print(gmdh.CriterionType.STABILITY)
#%%
""" ****************     ALGORITMO DE GRID SEARCH COM RIA (P26) ********************* """

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\Json\resultados_grid_search_ria.json'

# Define as combinações de hiperparâmetros
parametros_ria = {
    #'criterion': [gmdh.Criterion(gmdh.CriterionType.REGULARITY),gmdh.Criterion(gmdh.CriterionType.STABILITY)],
    #'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.STABILITY)], # aproximadamente 3min
    'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.SYM_REGULARITY),Criterion(CriterionType.STABILITY),Criterion(CriterionType.SYM_STABILITY),Criterion(CriterionType.UNBIASED_OUTPUTS),Criterion(CriterionType.SYM_UNBIASED_OUTPUTS),Criterion(CriterionType.UNBIASED_COEFFS),Criterion(CriterionType.ABSOLUTE_NOISE_IMMUNITY),Criterion(CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY)],
    #'criterion': [gmdh.Criterion(gmdh.CriterionType.REGULARITY)],
    #'criterion': [gmdh.Criterion(gmdh.CriterionType.STABILITY)], 
    
    'k_best': list(range(1, 26)),  # Valores ajustados de k_best para começar em 1
    'p_average': list(range(1, 16)),  # Valores ajustados de p_average para começar em 1
    'n_jobs': [-1],
    'test_size': [0.01, 0.25, 0.5, 0.75, 0.99],  # Valores ajustados de test_size
    
    #'polynomial_type':[gmdh.PolynomialType.QUADRATIC,gmdh.PolynomialType.LINEAR,gmdh.PolynomialType.LINEAR_COV]
    'polynomial_type':[gmdh.PolynomialType.QUADRATIC,gmdh.PolynomialType.LINEAR]
    #'polynomial_type':[gmdh.PolynomialType.QUADRATIC]
    #'polynomial_type':[gmdh.PolynomialType.LINEAR]
    #'polynomial_type':[gmdh.PolynomialType.LINEAR_COV]
}

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução grid search RIA: ', time_formatted)

# Gerar todas as combinações de hiperparâmetros
combinacoes = list(itertools.product(
    parametros_ria['criterion'],
    parametros_ria['k_best'],
    parametros_ria['p_average'],
    parametros_ria['n_jobs'],
    parametros_ria['test_size'],
    parametros_ria['polynomial_type']
))

# Loop por todas as combinações de hiperparâmetros
for i, (criterion, k_best, p_average, n_jobs, test_size,polynomial_type) in enumerate(combinacoes):
    try:
        # Treinando o modelo
        model = Ria()
        #model.fit(X_train, y_train, k_best=k_best, p_average=p_average, n_jobs=n_jobs, verbose=0, test_size=test_size,polynomial_type=polynomial_type)
        model.fit(X_train, y_train, criterion=criterion, k_best=k_best, p_average=p_average, n_jobs=n_jobs, verbose=0, test_size=test_size,polynomial_type=polynomial_type)
        
        # Fazendo previsões
        y_pred = model.predict(X_test)
        
        # Converter o array para uma string JSON (se necessário)
        # y_test_json = json.dumps(y_test.tolist())
        # y_pred_json = json.dumps(y_pred.tolist())

        # Armazenando avaliações
        # r2
        score = r2_score(y_test, y_pred)
        metrica_r2.append(score)
        
        # mse
        score = mean_squared_error(y_test, y_pred)
        rmse_calculado = np.sqrt(score)
        metrica_mse.append(score)
        
        # rmse
        metrica_rmse.append(rmse_calculado)
        
        # mape
        score = mape(y_test, y_pred)
        metrica_mape.append(score)
        
        # Armazenar ID da combinação
        id_resultado.append(serialize_hyperparameters(criterion, p_average, n_jobs, test_size, k_best, polynomial_type))
    except Exception as e:
        print(f"Erro com a combinação {criterion, p_average, n_jobs, test_size, k_best, polynomial_type}: {e}")

# Encontrar as melhores métricas
best_r2_index = np.argmax(metrica_r2)
best_mse_index = np.argmin(metrica_mse)
best_rmse_index = np.argmin(metrica_rmse)
best_mape_index = np.argmin(metrica_mape)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Obter os melhores resultados
best_results = {
    'Melhor R2': {
        'score': metrica_r2[best_r2_index],
        'hiperparametros': id_resultado[best_r2_index]
    },
    'Menor MSE': {
        'score': metrica_mse[best_mse_index],
        'hiperparametros': id_resultado[best_mse_index]
    },
    'Menor RMSE': {
        'score': metrica_rmse[best_rmse_index],
        'hiperparametros': id_resultado[best_rmse_index]
    },
    'Menor MAPE': {
        'score': metrica_mape[best_mape_index],
        'hiperparametros': id_resultado[best_mape_index]
    },
    'Estratificacao':{
        'Razao':estratificacao,
        'Seed': seed
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os melhores resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(best_results, file, indent=4)

# Imprimir os melhores resultados
print(f"Melhor R2: {metrica_r2[best_r2_index]} com hiperparâmetros {id_resultado[best_r2_index]}")
print(f"Menor MSE: {metrica_mse[best_mse_index]} com hiperparâmetros {id_resultado[best_mse_index]}")
print(f"Menor RMSE: {metrica_rmse[best_rmse_index]} com hiperparâmetros {id_resultado[best_rmse_index]}")
print(f"Menor MAPE: {metrica_mape[best_mape_index]} com hiperparâmetros {id_resultado[best_mape_index]}")
print(f"Tempo de Execucao: {execution_time} segundos")

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%
""" ****************     TESTE DE LEITURA DE ARQUIVO JSON ********************* """

# Definir o caminho do arquivo
input_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\resultados_grid_search_ria.json'

# Ler o arquivo JSON
with open(input_path, 'r') as file:
    best_results = json.load(file)

# Imprimir o conteúdo do arquivo JSON
print(json.dumps(best_results, indent=4))


#%%
""" ****************     ALGORITMO DE GRID SEARCH COM MULTI (P27) ********************* """

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\json\resultados_grid_search_multi.json'

# Define as combinações de hiperparâmetros
parametros_multi = {
    'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.SYM_REGULARITY),Criterion(CriterionType.STABILITY),Criterion(CriterionType.SYM_STABILITY),Criterion(CriterionType.UNBIASED_OUTPUTS),Criterion(CriterionType.SYM_UNBIASED_OUTPUTS),Criterion(CriterionType.UNBIASED_COEFFS),Criterion(CriterionType.ABSOLUTE_NOISE_IMMUNITY),Criterion(CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY)],
    'k_best': list(range(1, 26)),  # Valores ajustados de k_best para começar em 1
    'p_average': list(range(1, 16)),  # Valores ajustados de p_average para começar em 1
    'n_jobs': [-1],
    'test_size': [0.01, 0.25, 0.5, 0.75, 0.99],  # Valores ajustados de test_size     
}

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução grid search MULTI: ', time_formatted)

# Gerar todas as combinações de hiperparâmetros
combinacoes = list(itertools.product(
    parametros_multi['criterion'],
    parametros_multi['k_best'],
    parametros_multi['p_average'],
    parametros_multi['n_jobs'],
    parametros_multi['test_size']    
))

# Loop por todas as combinações de hiperparâmetros
for i, (criterion, k_best, p_average, n_jobs, test_size) in enumerate(combinacoes):
    try:
        # Treinando o modelo
        model = Multi()
        model.fit(X_train, y_train, criterion=criterion, k_best=k_best, p_average=p_average, n_jobs=n_jobs, verbose=0, test_size=test_size)
        
        # Fazendo previsões
        y_pred = model.predict(X_test)
        
        # Converter o array para uma string JSON (se necessário)
        # y_test_json = json.dumps(y_test.tolist())
        # y_pred_json = json.dumps(y_pred.tolist())

        # Armazenando avaliações
        # r2
        score = r2_score(y_test, y_pred)
        metrica_r2.append(score)
        
        # mse
        score = mean_squared_error(y_test, y_pred)
        rmse_calculado = np.sqrt(score)
        metrica_mse.append(score)
        
        # rmse
        metrica_rmse.append(rmse_calculado)
        
        # mape
        score = mape(y_test, y_pred)
        metrica_mape.append(score)
        
        # Armazenar ID da combinação
        id_resultado.append(serialize_hyperparameters(criterion,p_average, n_jobs,test_size,k_best))
    except Exception as e:
        print(f"Erro com a combinação {criterion,p_average, n_jobs,test_size,k_best}: {e}")

# Encontrar as melhores métricas
best_r2_index = np.argmax(metrica_r2)
best_mse_index = np.argmin(metrica_mse)
best_rmse_index = np.argmin(metrica_rmse)
best_mape_index = np.argmin(metrica_mape)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Obter os melhores resultados
best_results = {
    'Melhor R2': {
        'score': metrica_r2[best_r2_index],
        'hiperparametros': id_resultado[best_r2_index]
    },
    'Menor MSE': {
        'score': metrica_mse[best_mse_index],
        'hiperparametros': id_resultado[best_mse_index]
    },
    'Menor RMSE': {
        'score': metrica_rmse[best_rmse_index],
        'hiperparametros': id_resultado[best_rmse_index]
    },
    'Menor MAPE': {
        'score': metrica_mape[best_mape_index],
        'hiperparametros': id_resultado[best_mape_index]
    },
    'Estratificacao':{
        'Razao':estratificacao,
        'Seed': seed
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os melhores resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(best_results, file, indent=4)

# Imprimir os melhores resultados
print(f"Melhor R2: {metrica_r2[best_r2_index]} com hiperparâmetros {id_resultado[best_r2_index]}")
print(f"Menor MSE: {metrica_mse[best_mse_index]} com hiperparâmetros {id_resultado[best_mse_index]}")
print(f"Menor RMSE: {metrica_rmse[best_rmse_index]} com hiperparâmetros {id_resultado[best_rmse_index]}")
print(f"Menor MAPE: {metrica_mape[best_mape_index]} com hiperparâmetros {id_resultado[best_mape_index]}")
print(f"Tempo de Execucao: {execution_time} segundos")

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%

""" ****************     ALGORITMO DE GRID SEARCH COM COMBI (P28) ********************* """

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\Json\resultados_grid_search_combi.json'

# Define as combinações de hiperparâmetros
parametros_combi = {
    #'criterion': [gmdh.Criterion(gmdh.CriterionType.REGULARITY),gmdh.Criterion(gmdh.CriterionType.STABILITY)],
    'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.SYM_REGULARITY),Criterion(CriterionType.STABILITY),Criterion(CriterionType.SYM_STABILITY),Criterion(CriterionType.UNBIASED_OUTPUTS),Criterion(CriterionType.SYM_UNBIASED_OUTPUTS),Criterion(CriterionType.UNBIASED_COEFFS),Criterion(CriterionType.ABSOLUTE_NOISE_IMMUNITY),Criterion(CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY)],
    #'k_best': list(range(1, 26)),  # Valores ajustados de k_best para começar em 1
    'p_average': list(range(1, 16)),  # Valores ajustados de p_average para começar em 1
    'n_jobs': [-1],
    'test_size': [0.01, 0.25, 0.5, 0.75, 0.99],  # Valores ajustados de test_size 
    #'polynomial_type':[gmdh.PolynomialType.QUADRATIC,gmdh.PolynomialType.LINEAR,gmdh.PolynomialType.LINEAR_COV]
    #'polynomial_type':[gmdh.PolynomialType.QUADRATIC]
}

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução grid search COMBI: ', time_formatted)

# Gerar todas as combinações de hiperparâmetros
combinacoes = list(itertools.product(
    parametros_combi['criterion'],
    #parametros_multi['k_best'],
    parametros_combi['p_average'],
    parametros_combi['n_jobs'],
    parametros_combi['test_size']
    #,parametros_multi['polynomial_type']
))

# Loop por todas as combinações de hiperparâmetros
for i, (criterion, p_average, n_jobs, test_size) in enumerate(combinacoes):
    try:
        # Treinando o modelo
        model = Combi()
        model.fit(X_train, y_train, criterion=criterion, p_average=p_average, n_jobs=n_jobs, verbose=0, test_size=test_size)
        
        # Fazendo previsões
        y_pred = model.predict(X_test)
        
        # Converter o array para uma string JSON (se necessário)
        # y_test_json = json.dumps(y_test.tolist())
        # y_pred_json = json.dumps(y_pred.tolist())

        # Armazenando avaliações
        # r2
        score = r2_score(y_test, y_pred)
        metrica_r2.append(score)
        
        # mse
        score = mean_squared_error(y_test, y_pred)
        rmse_calculado = np.sqrt(score)
        metrica_mse.append(score)
        
        # rmse
        metrica_rmse.append(rmse_calculado)
        
        # mape
        score = mape(y_test, y_pred)
        metrica_mape.append(score)
        
        # Armazenar ID da combinação
        id_resultado.append(serialize_hyperparameters(criterion, p_average, n_jobs, test_size))
    except Exception as e:
        print(f"Erro com a combinação {criterion, p_average, n_jobs, test_size}: {e}")

# Encontrar as melhores métricas
best_r2_index = np.argmax(metrica_r2)
best_mse_index = np.argmin(metrica_mse)
best_rmse_index = np.argmin(metrica_rmse)
best_mape_index = np.argmin(metrica_mape)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Obter os melhores resultados
best_results = {
    'Melhor R2': {
        'score': metrica_r2[best_r2_index],
        'hiperparametros': id_resultado[best_r2_index]
    },
    'Menor MSE': {
        'score': metrica_mse[best_mse_index],
        'hiperparametros': id_resultado[best_mse_index]
    },
    'Menor RMSE': {
        'score': metrica_rmse[best_rmse_index],
        'hiperparametros': id_resultado[best_rmse_index]
    },
    'Menor MAPE': {
        'score': metrica_mape[best_mape_index],
        'hiperparametros': id_resultado[best_mape_index]
    },
    'Estratificacao':{
        'Razao':estratificacao,
        'Seed': seed
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os melhores resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(best_results, file, indent=4)

# Imprimir os melhores resultados
print(f"Melhor R2: {metrica_r2[best_r2_index]} com hiperparâmetros {id_resultado[best_r2_index]}")
print(f"Menor MSE: {metrica_mse[best_mse_index]} com hiperparâmetros {id_resultado[best_mse_index]}")
print(f"Menor RMSE: {metrica_rmse[best_rmse_index]} com hiperparâmetros {id_resultado[best_rmse_index]}")
print(f"Menor MAPE: {metrica_mape[best_mape_index]} com hiperparâmetros {id_resultado[best_mape_index]}")
print(f"Tempo de Execucao: {execution_time} segundos")

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%

""" ****************     ALGORITMO DE GRID SEARCH COM MIA (P29) ********************* """

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\Json\resultados_grid_search_mia.json'

# Define as combinações de hiperparâmetros
parametros_mia = {
    #'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.STABILITY)],#ERRO
    #'criterion': [gmdh.Criterion(gmdh.CriterionType.REGULARITY)],
    'criterion': [Criterion(CriterionType.REGULARITY)],# OK OK OK
    #'criterion': [Criterion(CriterionType.REGULARITY),Criterion(CriterionType.SYM_REGULARITY),Criterion(CriterionType.STABILITY),Criterion(CriterionType.SYM_STABILITY),Criterion(CriterionType.UNBIASED_OUTPUTS),Criterion(CriterionType.SYM_UNBIASED_OUTPUTS),Criterion(CriterionType.UNBIASED_COEFFS),Criterion(CriterionType.ABSOLUTE_NOISE_IMMUNITY),Criterion(CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY)],
    #'criterion': [Criterion(CriterionType.STABILITY)],
    'k_best': list(range(3, 6)),  # Valores ajustados de k_best para começar em 1
    'p_average': list(range(1, 16)),  # Valores ajustados de p_average para começar em 1
    'n_jobs': [-1],
    'test_size': [0.01, 0.25, 0.5, 0.75, 0.99],  # Valores ajustados de test_size 
    #'polynomial_type':[gmdh.PolynomialType.QUADRATIC,gmdh.PolynomialType.LINEAR,gmdh.PolynomialType.LINEAR_COV]
    'polynomial_type':[gmdh.PolynomialType.QUADRATIC,gmdh.PolynomialType.LINEAR]
}

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução grid search MIA: ', time_formatted)


# Gerar todas as combinações de hiperparâmetros
combinacoes = list(itertools.product(
    parametros_mia['criterion'],
    parametros_mia['k_best'],
    parametros_mia['p_average'],
    parametros_mia['n_jobs'],
    parametros_mia['test_size'],
    parametros_mia['polynomial_type']
))

# Loop por todas as combinações de hiperparâmetros
for i, (criterion, k_best, p_average, n_jobs, test_size,polynomial_type) in enumerate(combinacoes):
    try:
        # Treinando o modelo
        model = Mia()
        model.fit(X_train, y_train, criterion=criterion, k_best=k_best, p_average=p_average, n_jobs=n_jobs, verbose=0, test_size=test_size,polynomial_type=polynomial_type)
        
        # Fazendo previsões
        y_pred = model.predict(X_test)
        
        # Converter o array para uma string JSON (se necessário)
        # y_test_json = json.dumps(y_test.tolist())
        # y_pred_json = json.dumps(y_pred.tolist())

        # Armazenando avaliações
        # r2
        score = r2_score(y_test, y_pred)
        metrica_r2.append(score)
        
        # mse
        score = mean_squared_error(y_test, y_pred)
        rmse_calculado = np.sqrt(score)
        metrica_mse.append(score)
        
        # rmse
        metrica_rmse.append(rmse_calculado)
        
        # mape
        score = mape(y_test, y_pred)
        metrica_mape.append(score)
        
        # Armazenar ID da combinação
        id_resultado.append(serialize_hyperparameters(criterion, p_average, n_jobs, test_size, k_best, polynomial_type))
    except Exception as e:
        print(f"Erro com a combinação {criterion, p_average, n_jobs, test_size, k_best, polynomial_type}: {e}")        

# Encontrar as melhores métricas
best_r2_index = np.argmax(metrica_r2)
best_mse_index = np.argmin(metrica_mse)
best_rmse_index = np.argmin(metrica_rmse)
best_mape_index = np.argmin(metrica_mape)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Obter os melhores resultados
best_results = {
    'Melhor R2': {
        'score': metrica_r2[best_r2_index],
        'hiperparametros': id_resultado[best_r2_index]
    },
    'Menor MSE': {
        'score': metrica_mse[best_mse_index],
        'hiperparametros': id_resultado[best_mse_index]
    },
    'Menor RMSE': {
        'score': metrica_rmse[best_rmse_index],
        'hiperparametros': id_resultado[best_rmse_index]
    },
    'Menor MAPE': {
        'score': metrica_mape[best_mape_index],
        'hiperparametros': id_resultado[best_mape_index]
    },
    'Estratificacao':{
        'Razao':estratificacao,
        'Seed': seed
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os melhores resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(best_results, file, indent=4)

# Imprimir os melhores resultados
print(f"Melhor R2: {metrica_r2[best_r2_index]} com hiperparâmetros {id_resultado[best_r2_index]}")
print(f"Menor MSE: {metrica_mse[best_mse_index]} com hiperparâmetros {id_resultado[best_mse_index]}")
print(f"Menor RMSE: {metrica_rmse[best_rmse_index]} com hiperparâmetros {id_resultado[best_rmse_index]}")
print(f"Menor MAPE: {metrica_mape[best_mape_index]} com hiperparâmetros {id_resultado[best_mape_index]}")
print(f"Tempo de Execucao: {execution_time} segundos")

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%
"""### @ MODELO RIA (P15) - KFOLD COM MELHORES HIPERPARAMETROS  """

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\json\resultados_kfold_ria.json'

i = 0

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução k-fold RIA: ', time_formatted)

# Loop sobre os folds
for train_index, test_index in kf.split(X,y):                        
    
    #Divide os folds
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    # Treinando o modelo
    model = Ria()
    #model.fit(X_train_fold, y_train_fold,k_best=25, p_average=15,n_jobs=-1,verbose=0,test_size=0.4)
    #model.fit(X_train_fold, y_train_fold,k_best=5, p_average=1,n_jobs=-1,verbose=0,test_size=0.5)
    model.fit(X_train_fold, y_train_fold,criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY),k_best=4, p_average=1,n_jobs=-1,verbose=0,test_size=0.25,polynomial_type=gmdh.PolynomialType.QUADRATIC)
    #model.fit(np.array(X_train_fold), np.array(y_train_fold))
        
    # Fazendo previsões
    y_pred = model.predict(X_test_fold)
    
    #alimenta tabela de execuções no bd
    # Converter o array para uma string JSON
    #y_test_json = json.dumps(y_test_fold.tolist())
    #y_pred_json = json.dumps(y_pred.tolist())                            
    
    # Armazenando avaliações
    #r2
    score = r2_score(y_test_fold, y_pred)
    metrica_r2.append(score)    
    
    #mse
    score = mean_squared_error(y_test_fold, y_pred)
    rmse_calculado = np.sqrt(score)
    metrica_mse.append(score)

    #rmse
    metrica_rmse.append(rmse_calculado)

    #mape
    score =  mape(y_test_fold, y_pred)
    metrica_mape.append(score) 

    #id_resultado
    #id_resultado.append(valor_id_resultado)         

    i = i + 1  

# Calculando a média e desvio padrão dos scores
media_r2 = np.mean(metrica_r2)
desvio_padrao_r2 = np.std(metrica_r2)
media_mse = np.mean(metrica_mse)
desvio_padrao_mse = np.std(metrica_mse)
media_rmse = np.mean(metrica_rmse)
desvio_padrao_rmse = np.std(metrica_rmse)
media_mape = np.mean(metrica_mape)
desvio_padrao_mape = np.std(metrica_mape)

# Estrutura para armazenar todas as métricas e médias
resultados = {
    'metricas': {
        'R2': metrica_r2,
        'MSE': metrica_mse,
        'RMSE': metrica_rmse,
        'MAPE': metrica_mape
    },
    'medias': {
        'R2': media_r2,
        'Desvio R2': desvio_padrao_r2,
        'MSE': media_mse,
        'Desvio MSE': desvio_padrao_mse,
        'RMSE': media_rmse,
        'Desvio RMSE': desvio_padrao_rmse,
        'MAPE': media_mape,
        'Desvio MAPE': desvio_padrao_mape
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(resultados, file, indent=4)

# Imprimir as métricas e médias
print("Métricas RIA:")
print(pd.DataFrame(resultados['metricas']))

print("Médias RIA:")
print(pd.DataFrame(resultados['medias'], index=['Média']))
print(f"Tempo de Execucao: {execution_time} segundos")

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%

# Definir o caminho do arquivo
input_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\resultados_kfold_ria.json'

# Ler o arquivo JSON
with open(input_path, 'r') as file:
    resultados = json.load(file)

# Imprimir o conteúdo do arquivo JSON
print("Métricas RIA:")
print(pd.DataFrame(resultados['metricas']))

print("Médias RIA:")
print(pd.DataFrame(resultados['medias'], index=['Média']))


#%%
model.get_best_polynomial()

#%%


#%%
"""### MODELO COMBI (P16) """

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\json\resultados_kfold_combi.json'

i = 0

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução k-fold COMBI: ', time_formatted)

# Loop sobre os folds
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Treinando o modelo
    model = Combi()
    model.fit(X_train_fold, y_train_fold,criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY), p_average=1,n_jobs=-1,verbose=0,test_size=0.01)
    #model.fit(np.array(X_train_fold), np.array(y_train_fold))
        
    # Fazendo previsões
    y_pred = model.predict(X_test_fold)
    
    #alimenta tabela de execuções no bd
    # Converter o array para uma string JSON
    #y_test_json = json.dumps(y_test.tolist())
    #y_pred_json = json.dumps(y_pred.tolist())  
    

    # Armazenando avaliações
    #r2
    score = r2_score(y_test_fold, y_pred)
    metrica_r2.append(score)    
    
    #mse
    score = mean_squared_error(y_test_fold, y_pred)
    rmse_calculado = np.sqrt(score)
    metrica_mse.append(score)

    #rmse   
    metrica_rmse.append(rmse_calculado)

    #mape
    score =  mape(y_test_fold, y_pred)
    metrica_mape.append(score) 

    #id_resultado
    #id_resultado.append(valor_id_resultado)       
    
    # Guardando hiperparametros
    #inserir_dados(df_hiper_combi, valor_id_resultado)      

    i = i + 1    

# Calculando a média e desvio padrão dos scores
media_r2 = np.mean(metrica_r2)
desvio_padrao_r2 = np.std(metrica_r2)
media_mse = np.mean(metrica_mse)
desvio_padrao_mse = np.std(metrica_mse)
media_rmse = np.mean(metrica_rmse)
desvio_padrao_rmse = np.std(metrica_rmse)
media_mape = np.mean(metrica_mape)
desvio_padrao_mape = np.std(metrica_mape)

# Estrutura para armazenar todas as métricas e médias
resultados = {
    'metricas': {
        'R2': metrica_r2,
        'MSE': metrica_mse,
        'RMSE': metrica_rmse,
        'MAPE': metrica_mape
    },
    'medias': {
        'R2': media_r2,
        'Desvio R2': desvio_padrao_r2,
        'MSE': media_mse,
        'Desvio MSE': desvio_padrao_mse,
        'RMSE': media_rmse,
        'Desvio RMSE': desvio_padrao_rmse,
        'MAPE': media_mape,
        'Desvio MAPE': desvio_padrao_mape
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(resultados, file, indent=4)

# Imprimir as métricas e médias
print("Métricas COMBI:")
print(pd.DataFrame(resultados['metricas']))

print("Médias COMBI:")
print(pd.DataFrame(resultados['medias'], index=['Média']))
print(f"Tempo de Execucao: {execution_time} segundos")

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)


#%%
"""### MODELO MULTI (P17) """

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\Json\resultados_kfold_multi.json'

i = 0

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução k-fold MULTI: ', time_formatted)

# Loop sobre os folds
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Treinando o modelo
    model = Multi()
    model.fit(X_train_fold, y_train_fold, criterion=gmdh.Criterion(gmdh.CriterionType.STABILITY),k_best=1, p_average=1,n_jobs=-1,verbose=0,test_size=0.01)
    #model.fit(np.array(X_train_fold), np.array(y_train_fold))
        
    # Fazendo previsões
    y_pred = model.predict(X_test_fold)
    
    #alimenta tabela de execuções no bd
    # Converter o array para uma string JSON
    #y_test_json = json.dumps(y_test.tolist())
    #y_pred_json = json.dumps(y_pred.tolist())      

    # Armazenando avaliações
    #r2
    score = r2_score(y_test_fold, y_pred)
    metrica_r2.append(score)    
    
    #mse
    score = mean_squared_error(y_test_fold, y_pred)
    rmse_calculado = np.sqrt(score)
    metrica_mse.append(score)

    #rmse    
    metrica_rmse.append(rmse_calculado)

    #mape
    score =  mape(y_test_fold, y_pred)
    metrica_mape.append(score)        
    
    #id_resultado
    #id_resultado.append(valor_id_resultado) 

    # Guardando hiperparametros
    #inserir_dados(df_hiper_multi, valor_id_resultado) 
               

    i = i + 1    

# Calculando a média e desvio padrão dos scores
media_r2 = np.mean(metrica_r2)
desvio_padrao_r2 = np.std(metrica_r2)
media_mse = np.mean(metrica_mse)
desvio_padrao_mse = np.std(metrica_mse)
media_rmse = np.mean(metrica_rmse)
desvio_padrao_rmse = np.std(metrica_rmse)
media_mape = np.mean(metrica_mape)
desvio_padrao_mape = np.std(metrica_mape)

# Estrutura para armazenar todas as métricas e médias
resultados = {
    'metricas': {
        'R2': metrica_r2,
        'MSE': metrica_mse,
        'RMSE': metrica_rmse,
        'MAPE': metrica_mape
    },
    'medias': {
        'R2': media_r2,
        'Desvio R2': desvio_padrao_r2,
        'MSE': media_mse,
        'Desvio MSE': desvio_padrao_mse,
        'RMSE': media_rmse,
        'Desvio RMSE': desvio_padrao_rmse,
        'MAPE': media_mape,
        'Desvio MAPE': desvio_padrao_mape
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(resultados, file, indent=4)

# Imprimir as métricas e médias
print("Métricas MULTI:")
print(pd.DataFrame(resultados['metricas']))

print("Médias MULTI:")
print(pd.DataFrame(resultados['medias'], index=['Média']))
print(f"Tempo de Execucao: {execution_time} segundos")

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)

#%%
"""### MODELO MIA (P18) """

# Lista para armazenar os scores
metrica_r2 = []
metrica_mse = []
metrica_rmse = []
metrica_mape = []
id_resultado = []

# Definir o caminho do arquivo
output_path = r'E:\Vagner\DS\Mestrado\Dissertação\FBHP\Spyder\json\resultados_kfold_mia.json'

# Medir o tempo de execução
start_time = time.time()

#Exibindo hora de inicio da execução
timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)
time_formatted = dt_object.strftime('%H:%M:%S')
# Exibir o resultado
print('Início da execução k-fold MIA: ', time_formatted)

i = 0

# Loop sobre os folds
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Treinando o modelo
    model = Mia()
    model.fit(X_train_fold, y_train_fold,criterion=gmdh.Criterion(gmdh.CriterionType.REGULARITY),k_best=5, p_average=1,n_jobs=-1,verbose=0,test_size=0.5,polynomial_type=gmdh.PolynomialType.QUADRATIC)  
        
    # Fazendo previsões
    y_pred = model.predict(X_test_fold)
    
    #alimenta tabela de execuções no bd
    # Converter o array para uma string JSON
    #y_test_json = json.dumps(y_test.tolist())
    #y_pred_json = json.dumps(y_pred.tolist())      

    # Armazenando avaliações
    #r2
    score = r2_score(y_test_fold, y_pred)
    metrica_r2.append(score)    
    
    #mse
    score = mean_squared_error(y_test_fold, y_pred)
    rmse_calculado = np.sqrt(score)
    metrica_mse.append(score)

    #rmse    
    metrica_rmse.append(rmse_calculado)

    #mape
    score =  mape(y_test_fold, y_pred)
    metrica_mape.append(score)

    #id_resultado
    #id_resultado.append(valor_id_resultado)  

    # Guardando hiperparametros
    #inserir_dados(df_hiper_mia, valor_id_resultado)           

    i = i + 1    

# Calculando a média e desvio padrão dos scores
media_r2 = np.mean(metrica_r2)
desvio_padrao_r2 = np.std(metrica_r2)
media_mse = np.mean(metrica_mse)
desvio_padrao_mse = np.std(metrica_mse)
media_rmse = np.mean(metrica_rmse)
desvio_padrao_rmse = np.std(metrica_rmse)
media_mape = np.mean(metrica_mape)
desvio_padrao_mape = np.std(metrica_mape)

# Estrutura para armazenar todas as métricas e médias
resultados = {
    'metricas': {
        'R2': metrica_r2,
        'MSE': metrica_mse,
        'RMSE': metrica_rmse,
        'MAPE': metrica_mape
    },
    'medias': {
        'R2': media_r2,
        'Desvio R2': desvio_padrao_r2,
        'MSE': media_mse,
        'Desvio MSE': desvio_padrao_mse,
        'RMSE': media_rmse,
        'Desvio RMSE': desvio_padrao_rmse,
        'MAPE': media_mape,
        'Desvio MAPE': desvio_padrao_mape
    },
    'Tempo de Execucao (s)': execution_time
}

# Salvar os resultados em um arquivo JSON
with open(output_path, 'w') as file:
    json.dump(resultados, file, indent=4)

# Imprimir as métricas e médias
print("Métricas MIA:")
print(pd.DataFrame(resultados['metricas']))

print("Médias MIA:")
print(pd.DataFrame(resultados['medias'], index=['Média']))
print(f"Tempo de Execucao: {execution_time} segundos")

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Emitir aviso sonoro
frequency = 1000  # Frequência em Hertz
duration = 3000   # Duração em milissegundos (1 segundo)
winsound.Beep(frequency, duration)


#%%
""" ****************     Exibição das métricas (P19) ********************* """
print("Métricas RIA:")
print(df_metricas_ria)

print("Métricas COMBI:")
print(df_metricas_combi)

print("Métricas MULTI:")
print(df_metricas_multi)

print("Métricas MIA:")
print(df_metricas_mia)


#%%
df_metricas_ria                        

