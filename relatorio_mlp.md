# Regressão com MLP - Relatório dos Experimentos

## 1. O que foi implementado

O `MLPRegressor` foi implementado manualmente em NumPy (`codigo/modelos.py`), sem bibliotecas de aprendizado de máquina. A implementação inclui:

- Propagação direta (forward) com camadas ocultas configuráveis e saída linear (uma nota prevista).
- Retropropagação do erro (backpropagation) calculada à mão.
- Descida de gradiente em mini-lotes (tamanho 64), com opção de momento e regularização L2.
- Inicialização de He para ReLU e de Xavier para Tanh/Sigmoide.
- Padronização interna do alvo durante o treino, com retorno à escala original em `predict`.
- Funções de ativação: **ReLU**, **Tanh** e **Sigmoide**.
- Histórico de EQM por época, usado para a curva de aprendizado.

A validação é a mesma dos demais modelos: **5 folds**, padronização ajustada apenas no treino de cada fold (sem vazamento) e a mesma semente, o que torna os resultados comparáveis.

O experimento está em `codigo/experimento_mlp.py` e é executado automaticamente pelo `main.py`.

> **Nomes das métricas:** nas tabelas e gráficos, **EQM**, **REQM** e **EAM** são os nomes em português de *MSE* (erro quadrático médio), *RMSE* (raiz do erro quadrático médio) e *MAE* (erro absoluto médio). Os arquivos `resultados/*.csv` e `resultados/resultados.json` mantêm as chaves originais em inglês.

## 2. Topologias avaliadas

Cada configuração altera um fator por vez em relação à base `64-32 (ReLU)`.

| Topologia | Camadas | Ativação | LR | Épocas | Params | R² treino | R² teste | Gap R² | EQM teste | REQM teste | EAM teste | Treino (s) | Teste (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 (ReLU) | [16] | relu | 0.01 | 200 | 209 | 0.3895 | 0.3513 | 0.0382 | 0.4943 | 0.7029 | 0.5479 | 0.828 | 0.0002 |
| 32 (ReLU) | [32] | relu | 0.01 | 200 | 417 | 0.4021 | 0.3498 | 0.0522 | 0.4953 | 0.7037 | 0.5475 | 0.911 | 0.0003 |
| 64-32 (ReLU) | [64, 32] | relu | 0.01 | 200 | 2881 | 0.5449 | 0.3387 | 0.2062 | 0.5038 | 0.7097 | 0.5442 | 1.733 | 0.0011 |
| 128-64-32 (ReLU) | [128, 64, 32] | relu | 0.01 | 200 | 11905 | 0.6406 | 0.2344 | 0.4062 | 0.5834 | 0.7625 | 0.5813 | 4.932 | 0.0022 |
| 64-32 (Tanh) | [64, 32] | tanh | 0.01 | 200 | 2881 | 0.4625 | 0.3685 | 0.0940 | 0.4812 | 0.6936 | 0.5362 | 2.563 | 0.0021 |
| 64-32 (Sigmoide) | [64, 32] | sigmoid | 0.01 | 200 | 2881 | 0.3049 | 0.2973 | 0.0076 | 0.5355 | 0.7316 | 0.5658 | 2.674 | 0.0026 |
| 64-32 (lr=0.001) | [64, 32] | relu | 0.001 | 200 | 2881 | 0.3965 | 0.3509 | 0.0457 | 0.4946 | 0.7032 | 0.5452 | 1.787 | 0.0011 |
| 64-32 (lr=0.05) | [64, 32] | relu | 0.05 | 200 | 2881 | 0.6437 | 0.2838 | 0.3599 | 0.5456 | 0.7383 | 0.5634 | 1.790 | 0.0013 |
| 64-32 (400 épocas) | [64, 32] | relu | 0.01 | 400 | 2881 | 0.5248 | 0.1970 | 0.3278 | 0.6114 | 0.7774 | 0.5913 | 3.572 | 0.0013 |

## 3. Comparação com os demais modelos

| Modelo | R² | R² ajustado | EQM | REQM | EAM | Treino (s) | Teste (s) |
|---|---|---|---|---|---|---|---|
| kNN (Euclidiana) | 0.3584 | 0.3529 | 0.4888 | 0.6990 | 0.5261 | 0.0000 | 0.1816 |
| kNN (Manhattan) | 0.3588 | 0.3533 | 0.4885 | 0.6988 | 0.5217 | 0.0000 | 0.3212 |
| Regressão Linear Múltipla | 0.2889 | 0.2828 | 0.5419 | 0.7360 | 0.5696 | 0.0011 | 0.0001 |
| **MLP (64-32, Tanh)** | **0.3685** | **0.3631** | **0.4812** | **0.6936** | 0.5362 | 2.5626 | 0.0021 |

**Leitura:**

- O MLP com Tanh foi o melhor em **R², EQM e REQM**, superando levemente o kNN.
- O kNN Manhattan continua com o **melhor EAM** (0,5217) e a melhor taxa de acerto com tolerância de 1 ponto (~89,9%).
- O MLP Tanh acerta 56,55% com tolerância de 0,5 e 86,32% com tolerância de 1,0 — competitivo, mas abaixo do kNN na tolerância de 1 ponto.
- A Regressão Linear foi o modelo mais rápido de treinar, porém o de pior desempenho.

## 4. Tempo de treinamento e de teste

- **Treino:** cresce com o número de parâmetros e de épocas. A rede pequena (16 neurônios) treina em ~0,8 s por fold; a rede profunda (128-64-32) chega a ~4,9 s; a configuração Tanh 64-32 fica em ~2,6 s.
- **Teste:** para todas as MLPs é praticamente instantâneo (~0,0002 a 0,0026 s por fold), muito mais rápido que o kNN (~0,18 a 0,32 s).
- **Comparação conceitual:** o kNN é *lazy learner* (treino quase zero, teste caro); o MLP é *eager learner* (treino caro, teste barato). A Regressão Linear é barata nas duas fases.

## 5. Capacidade de generalização

O gap entre R² de treino e de teste mostra o grau de overfitting:

- **Redes pequenas (16 e 32 neurônios):** gap de 0,038 e 0,052 — generalização estável.
- **64-32 ReLU:** gap de 0,206 — já começa a memorizar o treino.
- **128-64-32:** gap de 0,406 e R² de teste de apenas 0,234 — overfitting severo; mais capacidade piorou o resultado.
- **Learning rate 0,05:** gap de 0,360 — passos grandes demais desestabilizam o treino.
- **400 épocas:** gap de 0,328 — treinar por mais tempo também levou a overfitting.
- **Tanh 64-32:** gap de 0,094, com o melhor R² de teste (0,3685) — bom equilíbrio.
- **Sigmoide:** menor gap (0,0076), mas R² de teste de 0,297 — a saturação da sigmoide limita o aprendizado (underfitting).
- **lr=0,001:** gap de 0,046 e R² de 0,351 — opção estável e eficiente.

**Conclusão da generalização:** a partir de certo ponto, aumentar camadas, neurônios, learning rate ou épocas faz o modelo decorar o treino e piorar no teste. A Tanh funcionou como uma regularização implícita (não linearidade suave), enquanto a ReLU profunda overfittou. Técnicas como *early stopping*, L2 e dropout poderiam reduzir esse efeito.

## 6. Impacto do custo computacional

| Configuração | Parâmetros | Treino (s) | R² teste | Custo-benefício |
|---|---|---|---|---|
| 16 (ReLU) | 209 | 0.828 | 0.3513 | Melhor eficiência: rápido e com boa generalização |
| 32 (ReLU) | 417 | 0.911 | 0.3498 | Similar ao 16, sem ganho relevante |
| 64-32 (Tanh) | 2881 | 2.563 | 0.3685 | Melhor desempenho, custo moderado |
| 64-32 (lr=0.001) | 2881 | 1.787 | 0.3509 | Boa estabilidade com custo menor que a Tanh |
| 128-64-32 | 11905 | 4.932 | 0.2344 | Pior custo-benefício: caro e overfitado |

O custo de treino cresce aproximadamente com o número de parâmetros, mas o retorno em desempenho não acompanha esse crescimento. A rede profunda tem ~57× mais parâmetros que a de 16 neurônios, treina ~6× mais devagar e ainda assim tem R² de teste menor.

## 7. Verificação de corretude da implementação

Para garantir que a rede não está apenas "rodando", mas implementada corretamente, o script `codigo/verificacao_mlp.py` roda testes independentes dos resultados do vinho:

| Teste | O que verifica | Resultado |
|---|---|---|
| Formato dos pesos | arquitetura realmente multicamadas | `128-64-32` gera 4 matrizes: (11,128), (128,64), (64,32), (32,1) e 4 vieses |
| Forward manual | `predict` = propagação calculada à mão | erro 0,0 |
| Gradiente numérico | retropropagação derivada corretamente | erro relativo máx. ≤ 3×10⁻¹⁰ (1, 2 e 3 camadas ocultas × ReLU/Tanh/Sigmoide, pesos e vieses) |
| XOR | a camada oculta é necessária | sem camada oculta EQM 0,2500; com 8 neurônios Tanh EQM < 10⁻⁴ |
| Função não linear | capacidade de aproximação | R² 0,768 (MLP) vs 0,037 (regressão linear) |
| Escala do alvo | padronização interna reversível | média prevista 7,499 vs real 7,500 |
| Semente | reprodutibilidade | mesma semente → previsões idênticas; semente diferente → diferentes |
| Métricas | consistência com o pipeline | R² recalculado à mão = `metricas_regressao` (diferença < 10⁻¹²) |
| Momento e L2 | opções funcionam | momento altera o treino; L2 reduz a norma dos pesos (4,95 → 3,33) |

O teste de gradiente é o mais importante: compara o gradiente calculado pela retropropagação com o gradiente numérico (diferenças centrais) em **cada peso e viés**. A diferença da ordem de 10⁻¹⁰ confirma que a regra da cadeia está correta, inclusive para várias camadas e para as três ativações — não é um ajuste empírico, é a derivada certa.

## 8. Limitações e observações metodológicas

- O **R² ajustado** usa `p = 11` (número de atributos de entrada) para manter a comparabilidade com kNN e Regressão Linear. Ele não usa o número de pesos da rede; se usasse, o MLP seria fortemente penalizado pelo tamanho da arquitetura.
- A melhor topologia foi escolhida pela **média de R² nos folds de teste**. Uma abordagem mais rigorosa reservaria um conjunto de validação separado para a escolha da arquitetura.
- Não foram usados *early stopping*, dropout ou regularização L2 no conjunto principal de experimentos. A implementação suporta L2 e momento, mas eles ficaram desligados para isolar o efeito das topologias.
- As previsões não foram limitadas ao intervalo [3, 9]; o modelo pode, em tese, prever valores fora da faixa de notas.
- A curva de perda registra a média corrente calculada durante cada época, antes das atualizações de cada lote. Por isso o último ponto não é exatamente o EQM do modelo final — a diferença verificada foi inferior a 1% —, o que não afeta o treino nem as métricas.
- Os tempos variam de execução para execução conforme a carga da máquina; as métricas de desempenho são determinísticas por causa da semente fixa.

## 9. Conclusão

- **Melhor desempenho:** MLP `64-32` com ativação **Tanh**, lr 0,01 e 200 épocas (R² 0,3685; REQM 0,6936; EQM 0,4812). Foi o melhor modelo de regressão avaliado, à frente do kNN e da Regressão Linear.
- **Melhor equilíbrio geral:** a mesma configuração Tanh 64-32, por unir o melhor resultado preditivo a um gap de generalização aceitável (0,094) e custo de treino moderado (~2,6 s por fold).
- **Melhor eficiência:** `16` neurônios com ReLU (R² 0,3513, treino ~0,8 s) ou `64-32` com lr=0,001 (R² 0,3509, gap 0,046), para quem prioriza custo e estabilidade.
- **Pior custo-benefício:** `128-64-32` (muitos parâmetros, overfitting severo) e `lr=0,05`/`400 épocas` (também overfitadas).
- **Nuance importante:** o MLP venceu em R²/REQM/EQM porque é treinado minimizando erro quadrático; o kNN Manhattan ainda tem o melhor EAM e a melhor tolerância de 1 ponto. A escolha do "melhor" modelo depende da métrica e do objetivo.
- **Trabalhos futuros:** *early stopping*, regularização L2/dropout e mais amostras nas notas extremas (3 e 9) para melhorar a generalização.
