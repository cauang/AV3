import time

import numpy as np

from dados import carregar_arff, padronizar
from metricas import metricas_regressao, rotular
from modelos import MLPRegressor
from validacao import validacao_cruzada

atributos, alvo, nomes, _ = carregar_arff("../wine_quality.arff")
indices_treino, indices_teste = next(iter(validacao_cruzada(alvo, 5, semente=42)))
treino, teste = padronizar(atributos[indices_treino], atributos[indices_teste])

modelo = MLPRegressor(
    camadas_ocultas=(64, 32),
    ativacao="tanh",
    taxa_aprendizado=0.01,
    epocas=200,
    tamanho_lote=64,
    semente=42,
)

inicio = time.perf_counter()
modelo.fit(treino, alvo[indices_treino])
tempo_treino = time.perf_counter() - inicio

inicio = time.perf_counter()
previsto = modelo.predict(teste)
tempo_teste = time.perf_counter() - inicio

resultado = metricas_regressao(alvo[indices_teste], previsto, atributos.shape[1])
print("metricas:", {rotular(k): round(float(v), 4) for k, v in resultado.items()})
print(f"tempo treino: {tempo_treino:.2f}s | tempo teste: {tempo_teste:.4f}s")
print("real   :", alvo[indices_teste][:5])
print("previsto:", np.round(previsto[:5], 2))
print("primeira camada:", modelo.pesos[0].shape, "| pesos treinados:", modelo.quantidade_de_parametros(11))
