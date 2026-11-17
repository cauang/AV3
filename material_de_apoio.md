# Material de apoio — AV3 (IA Computacional, UNIFOR)

Acompanha o arquivo `Analise de credito e Vinho - AV3.pptx` (24 slides). A professora disse que cada membro responde **duas perguntas** sobre: escolha dos hiperparâmetros, funcionamento dos algoritmos, interpretação dos resultados, diferenças entre as arquiteturas e análise de desempenho. A Parte 3 traz perguntas e respostas nesses cinco temas.

---

## Parte 1 — Resumo em 30 segundos

- **Objetivo:** comparar modelos em dois problemas, crédito (classificação, 1.000 clientes) e vinho (regressão, 6.497 vinhos), e explicar *por que* cada um se comporta como se comporta.
- **Novidades da AV3:** Perceptron Simples (classificação), MLP (regressão) e análise do impacto do Z-score.
- **Resultado:**
  - Crédito: Bayes é a escolha prática (pega 58% dos maus pagadores); kNN tem a maior acurácia (75,5%); o Perceptron oscila e não compete.
  - Vinho: MLP 64-32 Tanh (R² 0,368) empata com o kNN (0,358); redes maiores sofrem overfitting.
  - Z-score: necessário para kNN, Perceptron e MLP; neutro para Bayes e regressão linear.
- **Mensagem:** não existe modelo melhor em tudo; a escolha pesa desempenho, generalização, custo e o tipo de erro que mais dói.

Divisão sugerida: André (slides 1 a 6), Julia (7 a 10), João Victor (11 a 16), Cauan (17 a 24).

---

## Parte 2 — O que falar em cada slide

### Slide 1 — Capa
- Este é o trabalho da AV3, extensão da AV2: mesmos dados, agora com modelos neurais e análise do Z-score.

### Slide 2 — Introdução
- Mantivemos datasets, validação cruzada, métricas e tabelas da AV2.
- Acrescentamos três coisas: Perceptron, MLP e Z-score.
- Os resultados de kNN, Bayes e regressão linear servem de referência para comparar.

### Slide 3 — Metodologia
- Validação cruzada de 5 folds, **estratificada** na classificação (mantém 70% bons e 30% maus em cada fold).
- Z-score calculado **só no treino** de cada fold e aplicado ao teste, para não vazar informação.
- Perceptron: 4 taxas × 7 épocas. MLP: 9 topologias. Tudo implementado à mão, só com NumPy.
- **Classe positiva = mau pagador.** Sensibilidade (recall) = VP/(VP+FN); especificidade = VN/(VN+FP).
- Regressão: R², R² ajustado, MSE, RMSE e MAE.

### Slide 4 — Crédito (resultados da AV2)
- Referência: aprovar todos acerta 70,0%.
- kNN Euclidiano 75,5%, Bayes 74,4% e kNN Manhattan 74,2%: um ganho de 4 a 5 pontos sobre a referência.
- O Bayes tem o melhor F1 (0,696).

### Slide 5 — Onde os modelos acertam e erram
- kNN: acerta 91% dos bons, mas só 38% dos maus. Aprova 62% dos maus pagadores.
- Bayes: acerta 82% dos bons e 58% dos maus. Aprova 42% dos maus.
- Aprovar um mau pagador é o erro mais caro: pesa 5× mais que recusar um bom (documentação do German Credit).

### Slide 6 — Efeito do k
- O pico é em k = 5 (75,5%). k = 1 segue o ruído do vizinho mais próximo.
- As diferenças de 1 a 2 pontos cabem no desvio entre folds (±2 a 3).

### Slide 7 — Perceptron: regra e configuração
- É um neurônio: saída = degrau(w·x + b). Se errou, corrige: w ← w + taxa·(y − ŷ)·x.
- Testamos taxa de aprendizado 0,001; 0,01; 0,1 e 1,0, e épocas 1, 5, 10, 25, 50, 100 e 200: 28 configurações.
- **Pesos iniciais:** com zero, a taxa só reescala os pesos e as previsões não mudam (0,728 nas 4 taxas). Usamos pesos aleatórios pequenos para a taxa fazer efeito.

### Slide 8 — Acurácia vs. épocas
- A acurácia **sobe e desce**: o crédito não é linearmente separável, então o Perceptron nunca converge.
- Melhor caso 74,2% (taxa 0,1; 50 épocas). Pior 60,5% (taxa 1,0; 100 épocas).
- A taxa muda pouco; cada curva varia mais com as épocas.

### Slide 9 — Perceptron frente aos modelos da AV2
- Aqui entram sensibilidade e especificidade.
- Perceptron de melhor acurácia: especificidade 92,9% (acerta os bons), mas sensibilidade 30,7% (pega só 3 em 10 maus pagadores).
- Perceptron de melhor F1 (lr 0,01; 100 épocas): sensibilidade 68%, especificidade 65%, acurácia só 65,6%.
- Bayes é o mais equilibrado: sensibilidade 57,7% e especificidade 81,6%.

### Slide 10 — Matrizes de confusão do Perceptron
- Melhor acurácia: aprova 208 dos 300 maus e recusa só 50 dos 700 bons.
- Melhor F1: pega 204 dos 300 maus, mas recusa 248 bons.
- Mudar a configuração só **troca** um tipo de erro pelo outro. No custo 5:1: 1.090 contra 728.

### Slide 11 — Vinho: resultados incluindo o MLP
- Referência: prever sempre a média erra 0,69 ponto.
- MLP (64-32, Tanh): R² 0,368 ± 0,018. kNN: 0,358 ± 0,026. Regressão linear: 0,289.
- **MLP e kNN empatam**: a diferença cabe no desvio entre folds.
- Nenhum modelo passa de R² 0,37: a nota é subjetiva e a química explica só parte dela.

### Slide 12 — O que pesa na nota
- Coeficientes padronizados da regressão linear: álcool +0,32, acidez volátil −0,22, açúcar residual +0,21, densidade −0,17.
- Cuidado: açúcar e densidade são correlacionados, então os sinais opostos se compensam.

### Slide 13 — Quão perto da nota real
- 56% dos vinhos ficam a até 0,5 ponto da nota real; cerca de 90% a até 1 ponto.
- Os modelos erram mais nas notas raras (3, 4, 8 e 9).

### Slide 14 — MLP: as 9 topologias
- Partimos da base 64-32 com ReLU, taxa 0,01 e 200 épocas, e mudamos **um fator por vez**: neurônios, camadas, ativação, taxa e épocas.
- A tabela mostra parâmetros, R² de treino e de teste, gap (treino − teste), erro e tempos.

### Slide 15 — Treino × teste
- Gap mede overfitting.
- 16 e 32 neurônios: gap de 0,04 a 0,05, generalizam mas aprendem pouco.
- 128-64-32: 0,64 no treino e 0,23 no teste. Decorou o treino.
- Taxa 0,05 e 400 épocas elevam o gap para 0,36 e 0,33. Com 400 épocas o resultado é instável (0,197 ± 0,205).

### Slide 16 — Qual configuração equilibra melhor
- 64-32 com Tanh: melhor teste (0,368), gap de 0,09, cerca de 1 s para treinar e menos de 1 ms para testar.
- 16 (ReLU) é barata e estável, mas limitada. A 128-64-32 sofre overfitting e é a mais cara.

### Slide 17 — Z-score: por que e como
- Os atributos têm escalas muito diferentes: valor do crédito de 250 a 18.424, idade e prazo na casa das dezenas; no vinho, SO₂ total na casa das centenas e pH perto de 3.
- Quem usa distância ou gradiente é dominado pelos números grandes.
- Aplicamos só no treino de cada fold e só nas colunas numéricas (as colunas 0/1 ficam como estão).

### Slide 18 — Z-score no crédito
- kNN Euclidiano: 66,0% → 75,5%. Bayes: sem mudança.
- Perceptron: 36,5% → 73,2%. Sem Z-score ele chama quase todos de mau pagador.
- Repare que o F1 do Perceptron *caiu* (0,457 → 0,290): sem Z-score ele acerta muito "mau", o que infla o F1 da classe positiva mesmo errando a maioria.

### Slide 19 — Z-score no vinho
- MLP: R² 0,059 → 0,368 (o maior ganho). kNN Euclidiano: 0,159 → 0,358.
- Regressão linear: idêntico (0,289).

### Slide 20 — Análise crítica do Z-score
- Contribuiu para kNN, Perceptron e MLP (+8 a +37 pontos de acurácia; +0,18 a +0,31 de R²).
- Neutro para Bayes e regressão linear.
- Não resolve o Perceptron (F1 máximo 0,54). O valor do crédito é assimétrico (assimetria 1,95; 25 pontos com |z| > 3): log ou escalonamento robusto são alternativas que não testamos.

### Slide 21 — Tempo de execução
- kNN: treino instantâneo, teste caro (0,28 s no vinho), porque compara cada amostra com todo o treino.
- Bayes e regressão linear: teste quase gratuito.
- Perceptron treina em cerca de 0,03 s; MLP em cerca de 1 s e testa em menos de 1 ms.

### Slide 22 — Pontos fortes e limitações (AV2)
- Resumo dos três algoritmos clássicos: kNN (melhor acurácia, sensível à escala), Bayes (rápido, assume independência) e regressão linear (interpretável, só relações lineares).
- Perceptron e MLP estão cobertos nos slides 7 a 16.

### Slide 23 — Conclusões
- Crédito: Bayes ou kNN. Vinho: MLP pequeno com Tanh ou kNN (empatados).
- O Z-score foi decisivo para kNN, Perceptron e MLP.
- Limitações: um k, uma partição, e os mesmos folds para escolher e avaliar as configurações (resultado um pouco otimista); tintos e brancos misturados.

### Slide 24 — Referências
- Dados do OpenML (credit-g e wine_quality); Rosenblatt (Perceptron, 1958) e Rumelhart et al. (retropropagação, 1986).

---

## Parte 3 — Perguntas prováveis, por tema

### A) Escolha dos hiperparâmetros

1. **Por que essas taxas de aprendizado e épocas no Perceptron?**
   Cobrem várias ordens de grandeza (0,001 a 1,0) e de épocas (1 a 200), para ver o comportamento de poucas até muitas atualizações.

2. **Por que usar pesos iniciais aleatórios e não zero?**
   Com zero, a taxa só reescala os pesos e as previsões ficam idênticas (0,728 nas 4 taxas). Com pesos aleatórios pequenos, a taxa passa a ter efeito e dá para estudar o hiperparâmetro.

3. **Mais épocas sempre melhoram?**
   Não. No Perceptron a acurácia oscila (60,5% com 100 épocas e 73% com 50). No MLP, 400 épocas deram R² 0,197 contra 0,339 com 200 (mesma topologia 64-32 ReLU), e o resultado ficou instável.

4. **Qual o efeito da taxa de aprendizado no MLP?**
   Taxa 0,001: R² 0,351 e gap 0,046, aprende devagar e estável. Taxa 0,01: 0,339 e gap 0,206. Taxa 0,05: 0,284 e gap 0,360, ajusta demais o treino.

5. **Por que k = 5 no kNN?**
   O pico da acurácia no crédito foi em k = 5 (75,5%). Com k = 1 o modelo segue o ruído do vizinho mais próximo.

6. **Por que variar um fator por vez no MLP?**
   Assim sabemos qual mudança causou qual efeito. A desvantagem é que não testamos todas as combinações.

7. **Por que 5 folds?**
   Dá cinco estimativas para média e desvio, com tempo de execução viável.

### B) Funcionamento dos algoritmos

8. **Como o Perceptron aprende?**
   Calcula a soma ponderada w·x + b e aplica um degrau (0 ou 1). Se errou, corrige os pesos na direção do erro: w ← w + taxa·(y − ŷ)·x. Se acertou, não muda nada.

9. **Por que ele não converge no crédito?**
   Só para de corrigir quando acerta tudo, e isso exige classes separáveis por um hiperplano. No crédito as classes se sobrepõem, então ele continua corrigindo e oscilando.

10. **Como o MLP aprende?**
    Faz a propagação para frente pelas camadas (soma ponderada e ativação), calcula o erro quadrático e usa a **retropropagação** para obter o gradiente de cada peso. Atualiza os pesos por descida do gradiente em mini-lotes de 64 amostras. A saída é linear, porque é regressão.

11. **Qual o papel da função de ativação?**
    Introduz não linearidade. Sem ela, várias camadas equivaleriam a uma única transformação linear. Testamos ReLU, Tanh e Sigmoide.

12. **Como funciona o kNN?**
    Guarda o treino. Para cada amostra nova, acha os k mais próximos (distância Euclidiana ou Manhattan) e vota (classificação) ou tira a média (regressão). Não treina, mas testa devagar.

13. **Como funciona o Bayes univariado?**
    Escolhe a classe de maior probabilidade a posteriori, multiplicando a priori pelas probabilidades de cada atributo (gaussiana nos contínuos, frequência nos categóricos). Assume atributos independentes.

14. **Como funciona a regressão linear múltipla?**
    Ajusta ŷ = β₀ + β₁x₁ + … por mínimos quadrados: β = (XᵀX)⁻¹Xᵀy. Cada coeficiente mostra o efeito de um atributo.

15. **O que é o Z-score?**
    (valor − média) / desvio padrão, por atributo. Deixa todos com média 0 e desvio 1. Calculamos só no treino de cada fold, para não vazar informação do teste.

### C) Interpretação dos resultados

16. **75% de acurácia é bom no crédito?**
    A referência (aprovar todos) já acerta 70%. O ganho real é de uns 5 pontos, e a acurácia sozinha esconde que o kNN deixa passar 62% dos maus pagadores.

17. **O que são sensibilidade e especificidade aqui?**
    Sensibilidade (recall) é a fração dos maus pagadores que o modelo pega. Especificidade é a fração dos bons que ele reconhece. Precisamos das duas, porque um modelo pode ter boa acurácia recusando bons ou aprovando maus.

18. **R² de 0,37 é bom?**
    É modesto. A nota do vinho é subjetiva e a química explica só parte dela. Mesmo assim reduz o erro médio de 0,69 (prever a média) para cerca de 0,53.

19. **O MLP é melhor que o kNN no vinho?**
    Não de forma clara: 0,368 ± 0,018 contra 0,358 ± 0,026. A diferença cabe no desvio entre folds. Considere-os empatados.

20. **O que significa o desvio nas tabelas?**
    A variação da métrica entre os 5 folds. Se duas médias diferem menos que isso, não dá para afirmar que uma é melhor.

21. **Por que os modelos erram mais nas notas extremas?**
    Notas 3, 4, 8 e 9 são raras (as notas 5 e 6 são 77% dos dados). Os modelos aprendem a prever perto da média.

22. **O que a matriz de confusão do Perceptron mostra?**
    Que mudar a configuração só troca um erro pelo outro: a melhor acurácia aprova 208 maus e recusa 50 bons; a melhor F1 aprova 96 maus e recusa 248 bons.

### D) Diferenças entre as arquiteturas

23. **Qual a diferença entre Perceptron e MLP?**
    O Perceptron é um único neurônio com degrau e só separa classes linearmente. O MLP tem camadas ocultas com ativações não lineares, treina por retropropagação e capta relações não lineares. Aqui o Perceptron classifica e o MLP faz regressão.

24. **Por que a rede 128-64-32 foi pior que a 16?**
    Tem 11.905 parâmetros contra 209, para cerca de 5.200 amostras de treino por fold. Ajusta o ruído: R² 0,64 no treino e 0,23 no teste. A de 16 neurônios fica em 0,39 e 0,35.

25. **ReLU, Tanh ou Sigmoide: qual foi melhor e por quê?**
    Tanh teve o melhor teste (0,368), ReLU 0,339 e Sigmoide 0,297. A sigmoide também teve o menor R² de treino (0,305): subajustou, ou seja, aprendeu pouco com as mesmas épocas e taxa. Não testamos a causa, então o motivo é hipótese.

26. **Uma camada ou duas?**
    Com ReLU, 16 e 32 neurônios (uma camada) tiveram teste de 0,351 e 0,350, e a 64-32 (duas camadas) teve 0,339. Mais camadas aumentaram o gap sem melhorar o teste.

27. **Qual a diferença entre MLP e regressão linear?**
    A regressão linear só capta relações lineares (R² 0,289). O MLP, com ativações não lineares, capta curvas e interações (R² 0,368).

### E) Análise de desempenho

28. **Qual o melhor modelo?**
    Depende do critério. Crédito: Bayes se o erro caro é aprovar mau pagador; kNN se a acurácia bruta importa. Vinho: MLP 64-32 Tanh ou kNN, empatados.

29. **Como o custo computacional entra na decisão?**
    kNN não treina, mas testa em 0,28 s no vinho. O MLP leva cerca de 1 s para treinar e menos de 1 ms para testar. Se o treino for raro e a predição frequente, o MLP é mais barato; se o contrário, o kNN.

30. **Como vocês mediram a generalização?**
    Pelo gap entre R² de treino e de teste. Gap grande significa overfitting (a 128-64-32 tem 0,41).

31. **O Z-score ajudou todos os modelos?**
    Não. Foi decisivo para kNN (+8 a +9,5 pontos), Perceptron (+36,7) e MLP (+0,31 de R²). Para Bayes e regressão linear o resultado foi idêntico.

32. **Por que o Z-score não mudou Bayes e regressão linear?**
    O Bayes estima média e variância por atributo e classe, então a escala some. Na regressão linear, padronizar só reescala os coeficientes e as previsões são as mesmas.

33. **Quais as limitações do trabalho?**
    Escolhemos as melhores configurações nos mesmos folds usados na avaliação (resultado um pouco otimista); uma única semente e uma única partição; tintos e brancos misturados; não testamos log ou escalonamento robusto para o valor do crédito.
