# Classificação e Regressão

Trabalho da disciplina de Inteligência Artificial Computacional (UNIFOR) comparando modelos de classificação e de regressão em dois datasets.

## Datasets

- **Vinho (regressão):** https://www.openml.org/d/287
- **Análise de crédito (classificação):** https://www.openml.org/d/46365

## Integrantes

André Lima, Cauan Gomes, Julia Ferreira e João Victor.

## Como executar

Requisito: Python 3 e NumPy.

```bash
cd codigo
python main.py
```

O script treina e avalia todos os modelos, imprime as tabelas no terminal e grava os resultados em `resultados/`. A execução completa leva cerca de 2 minutos, principalmente por causa das 9 topologias de MLP avaliadas em 5 folds.

Para rodar apenas o Perceptron Simples (classificação de crédito, grade de taxa de aprendizado × épocas):

```bash
cd codigo
python3 experimento_perceptron.py
```

Para rodar apenas a análise do Z-score (com × sem normalização em todos os modelos):

```bash
cd codigo
python3 analise_zscore.py
```

Para rodar apenas os experimentos de MLP:

```bash
cd codigo
python experimento_mlp.py
```

Para verificar a corretude da implementação do MLP (gradiente numérico, XOR, determinismo, métricas):

```bash
cd codigo
python verificacao_mlp.py
```

Para um exemplo mínimo de uso direto da classe (1 fold, topologia `64-32` Tanh, com previsões na tela):

```bash
cd codigo
python exemplo_mlp.py
```

## Estrutura do código

- `codigo/main.py` — orquestra os experimentos, imprime tabelas e gera os resultados.
- `codigo/dados.py` — leitura dos arquivos ARFF, codificação one-hot e padronização.
- `codigo/modelos.py` — kNN (classificação e regressão), Bayes univariado, Regressão Linear Múltipla e **MLP implementado manualmente**.
- `codigo/experimento_perceptron.py` — Perceptron Simples: grade de taxas de aprendizado e épocas, matriz de confusão, sensibilidade e especificidade.
- `codigo/analise_zscore.py` — comparação de todos os modelos com e sem Z-score.
- `codigo/experimento_mlp.py` — topologias de MLP avaliadas e coleta de métricas/tempos.
- `codigo/verificacao_mlp.py` — testes de corretude do MLP (gradiente numérico, XOR, determinismo, métricas).
- `codigo/exemplo_mlp.py` — exemplo mínimo de uso direto do `MLPRegressor` em um fold.
- `codigo/metricas.py` — acurácia, precisão, recall, F1, MSE, RMSE, MAE, R² e R² ajustado.
- `codigo/validacao.py` — validação cruzada k-fold (com opção estratificada).
- `codigo/analises.py` — análises exploratórias dos dados de crédito e do vinho.
- `codigo/graficos.py` — geração dos gráficos em SVG.

## Modelos avaliados

### Classificação (crédito)

- kNN com distância Euclidiana
- kNN com distância Manhattan
- Bayesiano univariado
- **Perceptron Simples** (implementado manualmente; taxas 0,001/0,01/0,1/1,0 × épocas 1 a 200)

### Regressão (vinho)

- kNN com distância Euclidiana
- kNN com distância Manhattan
- Regressão Linear Múltipla
- **MLP (Perceptron de Múltiplas Camadas)** com diferentes topologias:
  - camadas ocultas: `(16,)`, `(32,)`, `(64, 32)`, `(128, 64, 32)`;
  - ativações: ReLU, Tanh e Sigmoide;
  - taxas de aprendizado: 0,001, 0,01 e 0,05;
  - épocas: 200 e 400.

## Métricas

- **Classificação:** acurácia, precisão, recall, F1-Score, tempo de treino e tempo de teste.
- **Regressão:** EQM (MSE), REQM (RMSE), EAM (MAE), R², R² ajustado, tempo de treino e tempo de teste. Nas tabelas e gráficos aparecem os nomes em português; os CSVs e o JSON mantêm as chaves originais em inglês.

## Resultados gerados

- `resultados/classificacao_perceptron.csv` — grade completa do Perceptron.
- `resultados/zscore_credito.csv` e `resultados/zscore_vinho.csv` — impacto do Z-score.
- `resultados/resultados.json` — todos os resultados consolidados.
- `resultados/classificacao_credito.csv` — métricas dos modelos de classificação.
- `resultados/regressao_vinho.csv` — comparação entre kNN, Regressão Linear e o melhor MLP.
- `resultados/regressao_mlp_topologias.csv` — todas as topologias de MLP avaliadas.
- `resultados/figuras/*.svg` — gráficos comparativos, curva de aprendizado do MLP, matriz de confusão, coeficientes etc.

A discussão completa dos experimentos de MLP está em [`relatorio_mlp.md`](relatorio_mlp.md).
