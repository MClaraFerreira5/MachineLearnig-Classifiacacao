# Classificação de Saúde Fetal com Machine Learning - 2ª VA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Concluído-green)

Este repositório contém o projeto prático referente à **Segunda Verificação de Aprendizagem (2ª VA)** da disciplina de **Inteligência Artificial**, ministrada pelo Prof. Anderson Cavalcanti na **Universidade Federal Rural de Pernambuco (UFRPE)** - Unidade Acadêmica de Belo Jardim.

## 🏥 Sobre o Projeto

O objetivo deste trabalho é comparar o desempenho de diferentes algoritmos de classificação supervisionada em um cenário de **diagnóstico médico**. O estudo foca na classificação da saúde fetal baseada em exames de **Cardiotocografia (CTG)**, visando auxiliar na identificação precoce de fetos em condição patológica.

### 🎯 Objetivos Específicos
1.  Comparar 5 algoritmos clássicos de Machine Learning.
2.  Aplicar **Validação Cruzada Estratificada** (10-fold) para garantir robustez estatística, especialmente dado o desbalanceamento das classes (muitos casos normais vs. poucos patológicos).
3.  Otimizar hiperparâmetros utilizando **GridSearch**.
4.  Analisar métricas críticas para a saúde (Precisão, Revocação e F1-Score), além da Acurácia.

## 📊 O Dataset: Fetal Health

Foi utilizado o conjunto de dados público **Fetal Health Classification**, disponível no Kaggle.

* **Fonte:** [Kaggle - Fetal Health Classification](https://www.kaggle.com/andrewmvd/fetal-health-classification)
* **Descrição:** O dataset contém 2.126 registros de exames de cardiotocografia.
* **Atributos (Features):** 21 variáveis numéricas, incluindo frequência cardíaca basal, acelerações, movimentos fetais e contrações uterinas.
* **Variável Alvo (`fetal_health`):**
    * `1.0`: Normal
    * `2.0`: Suspeito
    * `3.0`: Patológico

## 🛠️ Metodologia Experimental

O projeto foi desenvolvido em **Python** utilizando a biblioteca **Scikit-Learn**.

### 1. Algoritmos Avaliados
1.  **Árvore de Decisão** (Decision Tree)
2.  **K-Vizinhos Mais Próximos** (KNN)
3.  **Naive Bayes** (GaussianNB)
4.  **Regressão Logística**
5.  **Redes Neurais** (MLP Classifier)

### 2. Pipeline de Processamento
* **Pré-processamento:** Aplicação de `StandardScaler` para normalizar as escalas dos dados (crucial para o bom desempenho do KNN e da MLP).
* **Validação:** Utilização de **StratifiedKFold (k=10)**. A estratificação é essencial neste dataset para garantir que todas as dobras de teste contenham exemplos da classe "Patológico" (classe minoritária).
* **Otimização:** Busca de melhores parâmetros via `GridSearchCV` (mínimo de 3 combinações por modelo).

## 📂 Estrutura do Repositório

```text
MachineLearning-Classificacao/
│
├── data/                   # Arquivo CSV do dataset (fetal_health.csv)
│
├── notebooks/              # Código fonte (Jupyter Notebook) comentado
│   └── projeto_fetal_health.ipynb
│
├── docs/                   # Artigo (PDF) e Apresentação
│   └── Artigo_SBC_Grupo.pdf
│
├── README.md               # Documentação do projeto
└── requirements.txt        # Lista de dependências
````
## 👥 Autores

Trabalho desenvolvido pelos discentes:

* Gabriel Alves
* Maria Clara Ferreira
* Sabrina Gabriele
* Yann Keven

---
*UFRPE - UABJ | Dezembro de 2025*

