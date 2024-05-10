# -*- coding: utf-8 -*-
"""Dissertacao_Vagner.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c7dbcwAbXEjnwp2p6CcBaVE0Gwj-lWTY

# **Previsão FBHP**<br>
UFJF - PGMC - Abril/2023
Prof. Dr. Leonardo Golliat<br>

## Versionamento

*   v0: 24/04/2023: Vagner: Criação da estrutura do notebook

#**1. Definição do Problema**

**Formato**<br>
Dados tratados resultantes de ETL

**Tipo de dado**<br>
CSV

Nomes das colunas

1. **SN**: Número de série
2. **WHP**: Pressão na cabeça do poço (Wellhead Pressure) xxx
3. **WFR**: Taxa de fluxo de água (Water Flow Rate)
4. **OFR**: Taxa de fluxo de óleo (Oil Flow Rate)
5. **GFR**: Taxa de fluxo de gás (Gas Flow Rate)
6. **WPD**: Produção diária de água (Water Production Daily)
7. **API**: Índice de gravidade específica do petróleo (Oil API (American Petroleum Institute) gravity)
8. **ID**: Diâmetro interno do tubo (Inner Diameter)
9. **WBHT**: Temperatura no fundo do poço (Bottomhole Temperature)
11. **<font color="red">BHP</font>**: Pressão de fundo de poço em escoamento (Flowing Bottomhole Pressure)

#**2. Importação de bibliotecas ***
"""

import sys
print(sys.version)

#Importação de bibliotecas
import pylab as pl
import pandas as pd
import numpy as np
pl.style.use('ggplot')
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

! pip install sweetviz

import sweetviz as sv
sv.feature_config.FeatureConfig(force_num='2')

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

pip install gmdh

import gmdh

from gmdh import Combi, Multi, Mia, Ria, split_data
#COMBI, MULTI, MIA, RIA

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import json
import os

"""#**3. Aquisição e limpeza dos dados ***

## Aquisição
"""

#Montando drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#Trocando de diretório
#os.chdir("/content/drive/My Drive/Colab Notebooks")
#os.chdir("/content/drive/My Drive")
os.chdir("/content/drive/MyDrive/Colab Notebooks/Mestrado/Dissertacao")

#df = pd.read_csv('Ayoub_v2.csv')
df = pd.read_csv('Shammari.csv',sep=';')

df.shape

"""## Limpeza

Não houve necessidade de limpeza e/ou transformação.
Também não há linhas com valores nulos.
"""

#Verificando valores nulos
df.isnull().sum()

"""#**4. Análise Exploratória dos Dados**

###Visualização
"""

# Estatísticas básicas
summary_statistics = df.describe()

# Exibindo as estatísticas
print("Estatísticas básicas para cada coluna:")
print(summary_statistics)

"""Indicativo de necessidade de escalonamento (feature scaling)"""

#Visualizando as colunas do dataset
df.info()

#Tabela de freqüência da variável alvo
df['BHP'].value_counts()

# Visualizar a distribuição da coluna alvo "BHP"
plt.figure(figsize=(10, 6))
sns.histplot(df['BHP'], kde=True, bins=30, color='blue')
plt.title('Distribuição de BHP')
plt.xlabel('BHP')
plt.show()

df.head()

sv_report = sv.analyze(df)

sv_report.show_notebook()

"""###Análise de correlações

**Análise principal de correlação**</br>

Matriz de correlação entre todos os atributos numéricos, exceto o FBHP. Utilizando a biblioteca pandas para calcular a matriz de correlação e a seaborn para criar um mapa de calor para uma visualização mais clara.
"""

df.corr()

# Excluindo a coluna FBHP
#numeric_columns_except_fbhp = ['WHP', 'WFR', 'OFR', 'GFR', 'WPD', 'API', 'ID', 'BHT', 'WHT']

# Calculando a matriz de correlação
#correlation_matrix = df[numeric_columns_except_fbhp].corr()
correlation_matrix = df.corr()

# Criando um mapa de calor para visualização
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação entre Atributos')
plt.show()

# Calcula a matriz de correlação
correlation_matrix = df.corr()

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

"""Fluxo de gás e fluxo de óleo com alta correção

##Outliers
"""

# Seleciona apenas as colunas numéricas
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Define o limite para identificar outliers (por exemplo, usando o método IQR)
def identify_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

# Analisa cada coluna para identificar outliers
outliers_info = {}
for col in numeric_columns.columns:
    outliers = identify_outliers(numeric_columns[col])
    outliers_info[col] = {
        'total_outliers': outliers.sum(),
        'outliers_percent': (outliers.sum() / len(df)) * 100,
        'lower_bound': numeric_columns[col][outliers].min(),
        'upper_bound': numeric_columns[col][outliers].max()
    }

# Exibe o resultado da análise de outliers para cada coluna
for col, info in outliers_info.items():
    print(f"Coluna: {col}")
    print(f"Total de outliers: {info['total_outliers']}")
    print(f"Porcentagem de outliers: {info['outliers_percent']:.2f}%")
    print(f"Limite inferior para outliers: {info['lower_bound']:.2f}")
    print(f"Limite superior para outliers: {info['upper_bound']:.2f}")
    print("\n")

# Plota box plots para visualizar outliers (opcional)
plt.figure(figsize=(12, 6))
plt.title("Box Plots das Colunas com Outliers")
sns.boxplot(data=numeric_columns[outliers_info.keys()])
plt.xticks(rotation=90)
plt.show()

df.info()

import matplotlib.pyplot as plt

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

"""##Escalonamento *"""

# Exibindo as primeiras linhas do DataFrame antes da do escalonamento
print(df.head())
df_esc = df.copy()

# Selecionando apenas os atributos para a padronização (excluindo 'SN' e 'FBHP')
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

"""#**5. Separando treino e teste ***"""

# Excluindo o atributo 'BHP'
X = df_esc.drop('BHP', axis=1)
#X = df.drop('BHP', axis=1)

# Atribuindo o atributo 'FBHP' como a variável alvo
y = df_esc['BHP']
#y = df['BHP']

# Dividindo a base de dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibindo as primeiras linhas dos conjuntos de treinamento e teste
print("Conjunto de Treinamento:")
print(X_train.head())
print("\nConjunto de Teste:")
print(X_test.head())

# Exibindo as primeiras linhas das variáveis alvo dos conjuntos de treinamento e teste
print("\nVariável Alvo do Conjunto de Treinamento:")
print(y_train.head())
print("\nVariável Alvo do Conjunto de Teste:")
print(y_test.head())

#Comparando conjunto de treino e teste
df_sweetviz = sv.compare(
                 [X_train, "Training Data"],
                 [X_test , "Test Data"],
                 )

#Exibindo comparação de treino e teste
df_sweetviz.show_notebook()

"""#**6. GMDH**"""

x_train, x_test, y_train, y_test = split_data(X, y)

# print result arrays
print('x_train:\n', x_train)
print('x_test:\n', x_test)
print('\ny_train:\n', y_train)
print('y_test:\n', y_test)

#model = Combi()
#model = Multi()
#model = Mia()
model = Ria()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

# compare predicted and real value
print('y_predicted: ', y_predicted)
print('y_test: ', y_test)

#RASCUNHO *************************************
def fit_predict_mede_salva(modelo):
    #-----Fit/predict -------
    modelo.fit(x_train, y_train)
    y_predicted = modelo.predict(x_test)

    #-----Funções ----------
    def rmse(y_true, y_pred):
      return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    #-----Métricas ----------
    # Coeficiente de Determinação (R^2)
    r2 = r2_score(y_test, y_predicted)

    # Coeficiente de Pearson (R)
    pearson_corr, _ = pearsonr(y_test, y_predicted)

    # Erro Quadrático Médio (MSE)
    mse = mean_squared_error(y_test, y_predicted)

    # Erro Quadrático Médio Relativo (RMSE)
    root_mse = np.sqrt(mse)

    # Erro Percentual Médio Absoluto (MAPE)
    mape_val = mape(y_test, y_predicted)

    print("Modelo: ",modelo)
    print("Coeficiente de Determinação (R^2):", r2)
    print("Coeficiente de Pearson (R):", pearson_corr)
    print("Erro Quadrático Médio (MSE):", mse)
    print("Erro Quadrático Médio Relativo (RMSE):", root_mse)
    print("Erro Percentual Médio Absoluto (MAPE):", mape_val)

# RASCUNHO ***********************************
fit_predict_mede_salva(Ria())

"""#**7. Métricas**"""

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Coeficiente de Determinação (R^2)
r2 = r2_score(y_test, y_predicted)

# Coeficiente de Pearson (R)
pearson_corr, _ = pearsonr(y_test, y_predicted)

# Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_test, y_predicted)

# Erro Quadrático Médio Relativo (RMSE)
root_mse = np.sqrt(mse)

# Erro Percentual Médio Absoluto (MAPE)
mape_val = mape(y_test, y_predicted)

print("Coeficiente de Determinacao (R^2):", r2)
print("Coeficiente de Pearson (R):", pearson_corr)
print("Erro Quadratico Medio (MSE):", mse)
print("Erro Quadratico Medio Relativo (RMSE):", root_mse)
print("Erro Percentual Medio Absoluto (MAPE):", mape_val)

# Gravando JSON
metrics = {
    "Modelo: ": "RIA",
    "Coeficiente de Determinacao (R^2)": r2,
    "Coeficiente de Pearson (R)": pearson_corr,
    "Erro Quadratico Medio (MSE)": mse,
    "Erro Quadratico Medio Relativo (RMSE)": root_mse,
    "Erro Percentual Medio Absoluto (MAPE)": mape_val
}

# Nome do arquivo JSON
json_file = "metricas_gmdh_RIA.json"

# Salvar métricas em um arquivo JSON
with open(json_file, "w") as f:
    json.dump(metrics, f, indent=4)

print("As métricas foram salvas em:", json_file)

# RASCUNHO ***************************************************
# Nome do arquivo JSON
json_file = "metricas_gmdh.json"

# Verificar se o arquivo JSON já existe
if os.path.exists(json_file):
    # Carregar métricas do arquivo JSON existente
    with open(json_file, "r") as f:
        existing_metrics = json.load(f)
else:
    existing_metrics = {}

# Adicionar novas métricas ao dicionário existente
existing_metrics["Modelo: "] = "MULTI"
existing_metrics["Coeficiente de Determinacao (R^2)"] = r2
existing_metrics["Coeficiente de Pearson (R)"] = pearson_corr
existing_metrics["Erro Quadratico Medio (MSE)"] = mse
existing_metrics["Erro Quadratico Medio Relativo (RMSE)"] = root_mse
existing_metrics["Erro Percentual Medio Absoluto (MAPE)"] = mape_val

# Salvar métricas atualizadas em um arquivo JSON
with open(json_file, "w") as f:
    json.dump(existing_metrics, f, indent=4)

print("As métricas foram adicionadas ao arquivo:", json_file)

# RASCUNHO ***************************************************
# Lendo JSON
json_file = "metricas_gmdh.json"

# Abrir e ler o arquivo JSON
with open(json_file, "r") as f:
    metricas = json.load(f)

# Exibir as métricas lidas do arquivo JSON
print("Métricas lidas do arquivo", json_file, ":\n", metricas)

# Lendo JSON
json_file = "metricas_gmdh.json"

# Abrir e ler o arquivo JSON
with open(json_file, "r") as f:
    metricas = json.load(f)

# Criar DataFrame a partir das métricas
df = pd.DataFrame(metricas)

# Exibir DataFrame
print("Métricas:")
print(df)

# Exibir DataFrame em forma de tabela HTML
html_table = df.to_html()

# Exibir a tabela HTML
print(html_table)

"""<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COMBI</th>
      <th>MULTI</th>
      <th>MIA</th>
      <th>RIA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Coeficiente de Determinacao (R^2)</th>
      <td>0.637037</td>
      <td>0.637037</td>
      <td>-4.743916</td>
      <td>0.826511</td>
    </tr>
    <tr>
      <th>Coeficiente de Pearson (R)</th>
      <td>0.849399</td>
      <td>0.849399</td>
      <td>0.487659</td>
      <td>0.927486</td>
    </tr>
    <tr>
      <th>Erro Quadratico Medio (MSE)</th>
      <td>55713.350822</td>
      <td>55713.350822</td>
      <td>881668.656581</td>
      <td>26629.893973</td>
    </tr>
    <tr>
      <th>Erro Quadratico Medio Relativo (RMSE)</th>
      <td>236.036757</td>
      <td>236.036757</td>
      <td>938.972128</td>
      <td>163.186684</td>
    </tr>
    <tr>
      <th>Erro Percentual Medio Absoluto (MAPE)</th>
      <td>8.841795</td>
      <td>8.841795</td>
      <td>10.244377</td>
      <td>5.834186</td>
    </tr>
  </tbody>
</table>
"""

model.get_best_polynomial()