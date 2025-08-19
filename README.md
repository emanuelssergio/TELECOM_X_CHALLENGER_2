Projeto: Telecom X - Parte 2
Este projeto foi desenvolvido como a segunda etapa de um desafio de Ciência de Dados para a empresa fictícia Telecom X. O objetivo principal 
é construir modelos preditivos para identificar quais clientes têm maior probabilidade de cancelar seus serviços (churn), permitindo que a empresa atue de forma proativa na retenção.


## estrutura do Projeto##

├── challenger_2_teleconx .ipynb   # Notebook principal com toda a análise

├── TELECOM X dataframe.csv         # Dataset original

└── README.md  

## 3. Processo de Preparação dos Dados##

A etapa de preparação dos dados foi crucial para garantir que as informações estivessem prontas para a modelagem:

Classificação das Variáveis:

Variáveis Categóricas: Protecao_Dispositivo, Backup_Online, Suporte_Tecnico, Seguranca_Online, Streaming_Filmes e Streaming_TV.

Variáveis Numéricas: Cobranca_Mensal, Cobranca_Total, Meses_De_Contrato, entre outras.

Tratamento e Codificação:

Linhas com valores ausentes (NaN) foram removidas para garantir a integridade dos dados.

As variáveis categóricas foram transformadas em formato numérico utilizando a técnica de One-Hot Encoding.
Esta técnica cria uma nova coluna para cada categoria, convertendo-as em um formato binário (0 ou 1), o que as torna compatíveis com os algoritmos de Machine Learning.

Divisão em Conjuntos de Treino e Teste:

O conjunto de dados foi dividido em 70% para treino e 30% para teste (train_test_split). 
Esta etapa é fundamental para avaliar o desempenho do modelo em dados que ele não viu durante o treinamento, evitando o problema de overfitting.


## 4. Justificativas das Escolhas de Modelagem ##

Dois modelos de classificação foram escolhidos para este projeto, com o objetivo de comparar a performance de um modelo que não exige normalização com um que se beneficia dela.

Árvore de Decisão (DecisionTreeClassifier): Este modelo foi selecionado por sua capacidade de lidar com dados em diferentes escalas sem a necessidade de normalização prévia. 
Além disso, ele fornece a importância de cada variável, o que facilita a interpretação dos resultados e a identificação dos fatores mais influentes na evasão.

KNN (KNeighborsClassifier): Este modelo foi escolhido para comparação. O KNN é um algoritmo baseado em distância e, idealmente, requer a normalização dos dados. 


## 5. Gráficos e Insights Obtidos ##

A análise exploratória de dados (EDA) e a interpretação dos modelos trouxeram insights valiosos:

Análise de Correlação: A matriz de correlação revelou que o tipo de contrato mensal e o baixo número de meses de contrato são os 
fatores que apresentam a maior correlação positiva com a evasão.

Importância das Variáveis: O modelo de Árvore de Decisão e a análise de sensibilidade do AUC confirmaram que as variáveis mais 
relevantes para a previsão de churn são: Meses_De_Contrato, Tipo_Contrato_Month-to-month e o método de pagamento Electronic check.

Desempenho dos Modelos: O modelo de Árvore de Decisão superou o KNN em todas as métricas de desempenho, incluindo acurácia e AUC.

## 6. Como Executar o Notebook ##
   
Para rodar o notebook e reproduzir a análise, siga as instruções abaixo:

Pré-requisitos: Certifique-se de que você tem o Python 3 instalado.

Instalação de Bibliotecas: Instale as bibliotecas necessárias via pip.

BIBLIOTECAS UTILISADAS:

import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import cross_val_score

Carregamento dos Dados: O notebook carrega os dados diretamente de um URL público. Não é necessário fazer o download manual.

Execução: Abra o arquivo TELECOM X dataframe.csv.ipynb em um ambiente como Jupyter Notebook ou Google Colab e execute as células sequencialmente.


## AUTOR:
https://www.linkedin.com/in/emanuel-silva-sergio/
