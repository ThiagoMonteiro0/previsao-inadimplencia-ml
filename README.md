# previsao-inadimplencia-ml
Projeto de Machine Learning para previsão de inadimplência e análise de risco de crédito utilizando Python e ensemble learning.
# Sistema de Previsão de Inadimplência com Machine Learning

## Sobre o projeto

Este projeto foi desenvolvido com o objetivo de aplicar conceitos de Machine Learning em um problema real de análise de risco de crédito: a previsão de inadimplência de clientes.

A escolha do tema surgiu da minha experiência profissional atuando como analista em uma área de empréstimos consignados, onde tive contato com processos relacionados à análise de crédito, validação de informações financeiras e impacto do risco de inadimplência nas decisões de negócio.

Com base nessa experiência, o projeto foi criado buscando unir:

* conhecimento de negócio
* análise de dados
* Machine Learning aplicado
* interpretação de risco financeiro

---

# Objetivo

Construir um modelo capaz de prever a probabilidade de inadimplência de clientes com base em informações financeiras e histórico de pagamentos.

Além da classificação tradicional (`0` ou `1`), o projeto também busca simular um fluxo mais próximo de sistemas reais de crédito, utilizando:

* probabilidades
* faixas de risco
* regras de decisão
* revisão manual

---

# Base de dados utilizada

O projeto foi desenvolvido utilizando o dataset público:

[Credit Risk Dataset - Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data?utm_source=chatgpt.com)

O dataset contém informações relacionadas a:

* histórico de pagamentos
* valores de faturas
* pagamentos anteriores
* limite de crédito
* dados demográficos
* comportamento financeiro

A variável alvo utilizada no projeto representa a ocorrência de inadimplência no pagamento.

O conjunto de dados foi utilizado para fins educacionais e de estudo em Machine Learning aplicado a risco de crédito. ([Kaggle][1])

---

# Problema de negócio

Em operações de crédito, aprovar clientes inadimplentes pode gerar prejuízos financeiros, enquanto reprovar bons clientes pode causar perda de oportunidades de negócio.

Por isso, o projeto foi desenvolvido considerando o equilíbrio entre:

* identificação de clientes de risco
* redução de falsos positivos
* apoio à tomada de decisão

---

# Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* OpenPyXL
* XLRD

---

# Etapas do projeto

## 1. Análise exploratória dos dados (EDA)

Foram realizadas análises para:

* verificar dados nulos
* entender distribuição das classes
* identificar desbalanceamento
* validar qualidade dos dados

---

## 2. Preparação dos dados

Etapas aplicadas:

* separação entre variáveis de entrada e variável alvo
* divisão treino/teste
* normalização com `StandardScaler`

---

## 3. Modelagem

Foram treinados e comparados diferentes algoritmos de classificação:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

---

# Métricas avaliadas

O projeto foi desenvolvido considerando que accuracy isoladamente não representa bem problemas desbalanceados.

Por isso, foram analisadas:

* Precision
* Recall
* F1-Score
* Accuracy

Com foco principal na capacidade de identificar clientes inadimplentes.

---

# Trade-off entre métricas

Durante os testes foi possível observar comportamentos diferentes entre os modelos:

* modelos mais agressivos aumentavam o recall
* modelos mais conservadores aumentavam a precision

Esse comportamento foi utilizado para estudar o impacto das decisões de negócio relacionadas ao risco de crédito.

---

# Técnicas aplicadas

## Ensemble Learning

Foi implementado um Ensemble Voting combinando múltiplos modelos para gerar previsões mais equilibradas.

---

## Weighted Voting

Os modelos com melhor desempenho receberam maior peso no ensemble:

* Random Forest
* Gradient Boosting

---

## Threshold Tuning

As probabilidades geradas pelos modelos foram utilizadas para ajustar o threshold de decisão, buscando equilíbrio entre:

* precisão
* recall
* risco operacional

---

## Feature Selection

Foi realizada análise de importância das variáveis utilizando Random Forest para:

* identificar atributos mais relevantes
* remover colunas com baixa contribuição
* reduzir ruído no modelo

---

# Principais aprendizados

Durante o desenvolvimento do projeto foram aprofundados conceitos como:

* classificação supervisionada
* ensemble learning
* feature importance
* threshold tuning
* análise de risco
* interpretação de métricas
* impacto do desbalanceamento de classes
* tomada de decisão baseada em probabilidade

---

# Simulação de decisão de crédito

Além da previsão binária, o projeto implementa uma lógica simples de análise de risco:

| Probabilidade | Decisão        |
| ------------- | -------------- |
| Baixo risco   | APROVADO       |
| Médio risco   | REVISÃO MANUAL |
| Alto risco    | REPROVADO      |

Essa abordagem busca aproximar o projeto de cenários reais de análise de crédito.

---

# Resultado final

O ensemble final apresentou um equilíbrio entre:

* identificação de inadimplentes
* controle de falsos positivos
* estabilidade geral do modelo

Métricas finais aproximadas:

| Métrica   | Classe inadimplente |
| --------- | ------------------- |
| Precision | 54%                 |
| Recall    | 50%                 |
| F1-Score  | 52%                 |

---

# Próximos passos

Evoluções futuras planejadas:

* ROC Curve e AUC
* Hyperparameter Tuning
* Cross Validation
* API com FastAPI
* Dashboard com Streamlit ou Power BI
* Deploy do modelo
* Monitoramento de métricas

---

# Considerações finais

Este projeto foi desenvolvido como estudo prático de Machine Learning aplicado a risco de crédito, buscando não apenas treinar modelos, mas também compreender:

* comportamento dos algoritmos
* impacto das métricas
* interpretação de probabilidades
* decisões de negócio baseadas em dados

O foco principal foi desenvolver raciocínio analítico e entendimento aplicado de Machine Learning em um contexto próximo ao mercado financeiro.

[1]: https://www.kaggle.com/datasets/laotse/credit-risk-dataset?utm_source=chatgpt.com "Credit Risk Dataset"
