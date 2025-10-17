# README — Modelo de Detecção de Fraudes em Transações de Cartão de Crédito 💳

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP) para um sistema de detecção automática de fraudes em transações financeiras, utilizando técnicas de aprendizado de máquina supervisionado. A proposta consiste em analisar um grande volume de transações com cartão de crédito, identificar padrões comportamentais e classificar automaticamente cada operação como “fraude” ou “não fraude”.

Por se tratar de um problema real e de alta criticidade, a base de dados apresenta forte desbalanceamento de classes, o que demanda estratégias adequadas para evitar métricas ilusórias e garantir maior sensibilidade na detecção de fraudes.

O projeto utiliza o dataset público Credit Card Fraud Detection, disponibilizado no Kaggle, escolhido com base em três fatores principais:

* Relevância prática: a detecção de fraudes em cartões de crédito é um dos problemas mais clássicos e críticos no uso de machine learning, com impacto direto na segurança financeira.
* Desafios técnicos: trata-se de um conjunto com mais de 280 mil registros e fraudes representando menos de 0,2% das amostras, configurando um cenário realista de classificação assimétrica que permite avaliar a robustez dos modelos.
* Reprodutibilidade: por ser público e amplamente utilizado em pesquisas e competições, o acesso via Kaggle API garante que os experimentos possam ser reproduzidos de forma padronizada.

Além disso, a base passou por um pré-processamento utilizando Análise de Componentes Principais (PCA), que resultou nas variáveis V1–V28. Essa etapa reduz a dimensionalidade, preserva a privacidade dos dados e gera atributos ortogonais com boa separabilidade estatística, influenciando diretamente nas decisões de modelagem adotadas nas fases subsequentes do projeto.

## 🧠 Hipótese
Transações fraudulentas apresentam padrões estatísticos e numéricos distintos em relação às transações legítimas, o que possibilita que algoritmos supervisionados aprendam a diferenciar esses dois grupos com boa capacidade de generalização.

A hipótese central do projeto é que, ao combinar métricas adequadas (como F1-score e ROC-AUC), técnicas de balanceamento de dados (como SMOTE) e ajuste do threshold de decisão, é possível obter um modelo com alto recall para a classe minoritária, preservando uma precisão satisfatória.

Para estabelecer um baseline interpretável, foi utilizada Regressão Logística, um modelo linear simples e amplamente empregado em problemas de classificação binária. Além de permitir avaliar a separabilidade linear entre as classes, esse modelo serve como referência inicial para comparar métodos mais sofisticados.

Em seguida, foi testado o modelo Random Forest, escolhido por sua robustez, capacidade de capturar relações não lineares e bom desempenho em cenários complexos. Entre suas principais vantagens neste contexto, destacam-se:
* Resistência natural a ruídos e outliers;
* Bom desempenho com dados desbalanceados, especialmente quando combinado com técnicas de oversampling e ajuste de threshold;
* Alta flexibilidade, com possibilidade de ajuste fino via hiperparâmetros e fornecimento de probabilidades calibradas para otimização do ponto de corte.

## 1) Configuração Inicial e Instalação de Pacotes
Nesta etapa foram instaladas e importadas todas as bibliotecas necessárias:

* kaggle — para download automatizado do dataset
* pandas/numpy/seaborn/matplotlib — para manipulação e análise exploratória
* scikit-learn — para modelagem e métricas
* imbalanced-learn — para balanceamento com SMOTE.

```python
!pip -q install kaggle imbalanced-learn

import os, json, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
import joblib
```

## 2) 📥 Download do Dataset via Kaggle

As credenciais da conta Kaggle foram configuradas diretamente no ambiente Colab, e o dataset foi baixado e descompactado automaticamente.
O arquivo CSV foi lido com pandas e apresentou 284.807 linhas e 31 colunas, sem valores ausentes.

```python
DATASET_SLUG = "mlg-ulb/creditcardfraud"
df = pd.read_csv("creditcard.csv")
print(df.shape)
```

## 3) 📊 Análise Exploratória dos Dados (EDA)
A etapa inicial de análise exploratória de dados (EDA) teve como objetivo compreender a estrutura e as principais características do dataset antes da etapa de modelagem. Primeiramente, verificou-se que não havia valores ausentes em nenhuma das 31 colunas, o que eliminou a necessidade de estratégias de imputação ou limpeza complexa. Além disso, observou-se que as variáveis V1 a V28 já se encontram padronizadas, resultado de uma transformação prévia por Análise de Componentes Principais (PCA), técnica comumente utilizada para anonimizar dados de cartão de crédito e, ao mesmo tempo, reduzir dimensionalidade preservando informações relevantes para detecção de padrões.

O histograma do Amount evidenciou uma distribuição altamente assimétrica, indicando que poucas transações concentram valores muito altos. Essa característica justificou a aplicação de uma transformação logarítmica log(Amount + 1), que suavizou a cauda longa e permitiu uma visualização mais clara da distribuição.

A matriz de correlação foi utilizada para avaliar possíveis multicolinearidades entre as variáveis — as componentes PCA apresentaram correlações relativamente baixas e bem distribuídas, indicando ausência de colinearidade excessiva.

Por fim, a análise da variável alvo Class mostrou um desbalanceamento extremo entre classes, evidenciando a necessidade de técnicas específicas de balanceamento para evitar que modelos enviesem predições para a classe majoritária.

Gráficos utilizados:
* Histograma de Amount
* Histograma de log(Amount+1)
* Matriz de correlação

## 4) ✂️ Divisão Treino/Teste (Stratified)

Os dados foram divididos em 80% treino e 20% teste, preservando a proporção das classes (stratify=y). Essa escolha garante representatividade da classe fraudulenta em ambos os conjuntos, evitando problemas de viés no treinamento ou avaliação.

## 5) 🧪 Modelagem Inicial — Baseline, Logistic Regression e Random Forest
Para estabelecer um ponto de partida comparativo, foram treinados três modelos:
Baseline com DummyClassifier:
* Accuracy ≈ 0.998
* Recall = 0 (não detectou nenhuma fraude).

Logistic Regression (com class_weight='balanced'):
* Recall alto, porém precisão baixa.

Random Forest:
* Melhor equilíbrio entre precisão e recall.
* F1-score muito superior ao baseline.
Essa etapa inicial confirmou que métricas de acurácia isoladas são enganosas em cenários desbalanceados e que modelos mais robustos, como Random Forest, oferecem desempenho mais consistente.

## 6) 📈 Verificação de Overfitting e Underfitting
As acurácias no treino e teste foram praticamente iguais para Random Forest, indicando boa generalização.
```python
Acurácia Treino: 1.0000  
Acurácia Teste:  0.9995
```
Overfitting e underfitting foram verificados comparando as métricas entre treino e teste e pela estabilidade na validação cruzada. Como as acurácias foram muito próximas e não houve degradação severa de métricas, não há sinais de overfitting. A ausência de underfitting indica que a capacidade do modelo foi suficiente para aprender padrões relevantes sem ficar limitada.

## 7) 🔹 Validação Cruzada (3 folds)
Foi aplicada validação cruzada estratificada com 3 folds para avaliar a consistência das métricas entre diferentes partições do conjunto de dados.
A estratificação garante que cada fold mantenha a proporção original das classes, fundamental em cenários desbalanceados. A escolha de 3 folds foi feita para equilibrar robustez com custo computacional.

O desvio padrão reduzido dos F1-scores do Random Forest indicou boa estabilidade e baixa variabilidade, reforçando a confiabilidade do modelo.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores_log = cross_val_score(pipe_log, X_train, y_train, cv=cv, scoring='f1')
scores_rf = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='f1')
```
## 8) 🧭 Ajuste de Hiperparâmetros — GridSearchCV
Em seguida, foi realizada a etapa de ajuste de hiperparâmetros por meio do GridSearchCV, utilizando também validação cruzada. O objetivo foi buscar a melhor combinação entre os parâmetros n_estimators (50 e 100) e max_depth (10 e None). Essa etapa permitiu encontrar um modelo Random Forest otimizado, que manteve alto desempenho no conjunto de teste, equilibrando capacidade de detecção de fraudes e generalização.
```python
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [10, None]
}

grid = GridSearchCV(pipe_rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_sample, y_train_sample)
best_model = grid.best_estimator_
```

## 9) 📉 Avaliação do Modelo Otimizado
Após a otimização, foi feita a avaliação detalhada do modelo ajustado. As métricas de desempenho mostraram Acurácia ≈ 0.9990, F1-score ≈ 0.6093 e ROC-AUC ≈ 0.9428, valores bastante altos, mas ainda com espaço para melhoria no recall da classe minoritária. Foram geradas duas visualizações essenciais para interpretação do modelo: a matriz de confusão, que evidencia a proporção de acertos e erros entre as classes, e a curva ROC, que ilustra a relação entre taxa de verdadeiros positivos e falsos positivos.

Para lidar mais diretamente com o forte desbalanceamento da variável alvo, foi introduzida a primeira estratégia de melhoria: rebalanceamento com SMOTE (Synthetic Minority Over-sampling Technique). Essa técnica gera novas amostras sintéticas da classe minoritária, equilibrando a distribuição de classes no conjunto de treino.
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

## 10) 🆕 Melhoria 1 — Balanceamento com SMOTE
O SMOTE (Synthetic Minority Over-sampling Technique) foi aplicado no conjunto de treino para lidar com o desbalanceamento de classes.
O SMOTE foi escolhido em vez de técnicas de undersampling porque mantém a diversidade da classe majoritária, permitindo melhor generalização.
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```
O modelo Random Forest reentreinado apresentou F1-score mais alto e maior recall, confirmando a efetividade do balanceamento.

## 11) 🆕 Melhoria 2 — Ajuste de Threshold de Decisão
A segunda melhoria aplicada foi o ajuste do limiar de decisão (threshold). O threshold padrão de 0.5 foi substituído por um valor ótimo determinado a partir da maximização do F1-score na curva Precision-Recall, permitindo um controle mais fino sobre a taxa de verdadeiros positivos e falsos positivos.
Essa abordagem aumentou significativamente a sensibilidade do modelo, reduzindo falsos negativos e melhorando o equilíbrio entre precisão e recall — dois indicadores críticos em problemas de detecção de fraude.

## 12) 📊 Comparação de Modelos
Com todas as etapas de modelagem concluídas, foi realizada uma comparação consolidada entre os modelos. Os resultados evidenciaram que a acurácia isolada pode ser enganosa, pois até mesmo um classificador que sempre prediz a classe majoritária obtém valores elevados nessa métrica. Em contrapartida, métricas mais adequadas para cenários desbalanceados, como F1-score e ROC-AUC, mostraram que o uso combinado de SMOTE e ajuste de threshold foi decisivo para elevar o desempenho dos modelos, sobretudo na detecção da classe minoritária (fraude).

Após todas as etapas de experimentação e otimização, o modelo com melhor performance foi a Random Forest otimizada via GridSearchCV, combinada com as estratégias de rebalanceamento com SMOTE e ajuste de threshold.

Essa solução destacou-se por:
* Alcançar alta capacidade de detecção de fraudes, apresentando recall e F1-score significativamente superiores aos demais modelos;
* Demonstrar boa capacidade de generalização, com métricas consistentes entre treino e teste e baixa variância na validação cruzada;
* Oferecer robustez prática, refletida em um ROC-AUC elevado e estabilidade mesmo diante de forte desbalanceamento de classes.

Assim, o modelo final atende plenamente ao objetivo central do projeto: maximizar a detecção de fraudes com o menor número possível de falsos negativos, preservando ao mesmo tempo uma boa precisão global.

## 13) 📊 Análise do Desbalanceamento
A análise do desbalanceamento foi complementada com um gráfico de barras, que ilustrou visualmente a disparidade extrema entre as classes “Não Fraude” e “Fraude”. Essa etapa reforçou a importância das técnicas de balanceamento aplicadas para que o modelo conseguisse aprender padrões da minoria de forma eficaz.

## 14) 📝 Resumo Técnico Final
* Dataset altamente desbalanceado.
* Métricas principais: F1-score e ROC-AUC.
* Técnicas aplicadas: validação cruzada, GridSearchCV, SMOTE e ajuste de threshold.
* Importância das features foi avaliada, e todas foram mantidas, pois contribuíram de forma complementar.
* Melhor modelo: Random Forest otimizado + SMOTE + threshold tuning.
* O modelo apresentou bom desempenho geral, estabilidade entre treino/teste e baixo risco de overfitting.

A escolha dessa combinação se justifica por oferecer o melhor equilíbrio entre recall e precisão, fundamental em cenários de fraude.

## 15) 💾 Salvamento do Modelo Final
Por fim, o modelo final foi salvo em arquivo .pkl para uso posterior ou integração em um pipeline de produção:
```python
import joblib
joblib.dump(best_model, 'best_model_creditcard_rf.pkl')
```

Como próximos passos recomendados, sugere-se explorar modelos mais avançados, como XGBoost e LightGBM, realizar calibração de probabilidades para melhor interpretação dos scores, implementar monitoramento de drift e explicabilidade (por exemplo, com SHAP ou LIME) e, futuramente, desenvolver um dashboard em tempo real para acompanhamento contínuo da performance do modelo em produção.




