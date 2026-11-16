"""Experimentos com o Perceptron de Múltiplas Camadas (MLP) para a regressão do vinho.

Cada topologia é avaliada com a mesma validação cruzada 5-fold usada pelos demais
modelos. Para cada fold, a padronização é ajustada apenas no treino (sem vazamento)
e a rede é treinada do zero. As métricas de treino e de teste são registradas para
permitir a discussão sobre generalização.
"""
import time

import numpy as np

from dados import padronizar
from metricas import metricas_regressao
from modelos import MLPRegressor

SEMENTE = 42
TAMANHO_LOTE = 64

# Cada topologia altera um fator por vez em relação à base "64-32 (ReLU)".
TOPOLOGIAS = [
    {"nome": "16 (ReLU)", "rotulo": "16", "camadas": (16,), "ativacao": "relu", "taxa": 0.01, "epocas": 200},
    {"nome": "32 (ReLU)", "rotulo": "32", "camadas": (32,), "ativacao": "relu", "taxa": 0.01, "epocas": 200},
    {"nome": "64-32 (ReLU)", "rotulo": "64-32", "camadas": (64, 32), "ativacao": "relu", "taxa": 0.01, "epocas": 200},
    {"nome": "128-64-32 (ReLU)", "rotulo": "128-64-32", "camadas": (128, 64, 32), "ativacao": "relu", "taxa": 0.01, "epocas": 200},
    {"nome": "64-32 (Tanh)", "rotulo": "Tanh", "camadas": (64, 32), "ativacao": "tanh", "taxa": 0.01, "epocas": 200},
    {"nome": "64-32 (Sigmoide)", "rotulo": "Sigmoide", "camadas": (64, 32), "ativacao": "sigmoid", "taxa": 0.01, "epocas": 200},
    {"nome": "64-32 (lr=0.001)", "rotulo": "lr 0.001", "camadas": (64, 32), "ativacao": "relu", "taxa": 0.001, "epocas": 200},
    {"nome": "64-32 (lr=0.05)", "rotulo": "lr 0.05", "camadas": (64, 32), "ativacao": "relu", "taxa": 0.05, "epocas": 200},
    {"nome": "64-32 (400 épocas)", "rotulo": "400 ép.", "camadas": (64, 32), "ativacao": "relu", "taxa": 0.01, "epocas": 400},
]


def contar_parametros(camadas_ocultas, quantidade_entradas):
    tamanhos = [quantidade_entradas] + list(camadas_ocultas) + [1]
    return sum(entrada * saida + saida for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]))


def _media_desvio(valores):
    return {"media": float(np.mean(valores)), "desvio": float(np.std(valores, ddof=1))}


def _resumir(metricas_por_fold):
    return {nome: _media_desvio(valores) for nome, valores in metricas_por_fold.items()}


def avaliar_topologias_mlp(atributos, alvo, folds, colunas_numericas=None, semente=SEMENTE):
    """Treina e avalia cada topologia; devolve métricas, previsões e históricos."""
    resultados, previsoes, historicos = {}, {}, {}

    for config in TOPOLOGIAS:
        nome = config["nome"]
        metricas_treino = {}
        metricas_teste = {}
        tempos_treino, tempos_teste = [], []
        previsto = np.zeros(len(alvo))
        historico = None

        for indices_treino, indices_teste in folds:
            treino, teste = padronizar(
                atributos[indices_treino], atributos[indices_teste], colunas_numericas
            )
            alvo_treino, alvo_teste = alvo[indices_treino], alvo[indices_teste]

            modelo = MLPRegressor(
                camadas_ocultas=config["camadas"],
                ativacao=config["ativacao"],
                taxa_aprendizado=config["taxa"],
                epocas=config["epocas"],
                tamanho_lote=TAMANHO_LOTE,
                semente=semente,
            )

            inicio = time.perf_counter()
            modelo.fit(treino, alvo_treino)
            tempos_treino.append(time.perf_counter() - inicio)

            inicio = time.perf_counter()
            previsto_teste = modelo.predict(teste)
            tempos_teste.append(time.perf_counter() - inicio)
            previsto_treino = modelo.predict(treino)

            for metrica, valor in metricas_regressao(
                alvo_treino, previsto_treino, atributos.shape[1]
            ).items():
                metricas_treino.setdefault(metrica, []).append(valor)
            for metrica, valor in metricas_regressao(
                alvo_teste, previsto_teste, atributos.shape[1]
            ).items():
                metricas_teste.setdefault(metrica, []).append(valor)

            previsto[indices_teste] = previsto_teste
            if historico is None:  # guarda a curva do primeiro fold
                historico = list(modelo.historico_perda)

        resumo_treino = _resumir(metricas_treino)
        resumo_teste = _resumir(metricas_teste)
        resumo_teste["Tempo Treino (s)"] = _media_desvio(tempos_treino)
        resumo_teste["Tempo Teste (s)"] = _media_desvio(tempos_teste)

        resultados[nome] = {
            "config": config,
            "parametros": contar_parametros(config["camadas"], atributos.shape[1]),
            "treino": resumo_treino,
            "teste": resumo_teste,
            "tempos_treino": tempos_treino,
            "tempos_teste": tempos_teste,
        }
        previsoes[nome] = previsto
        historicos[nome] = historico

    return {"resultados": resultados, "previsoes": previsoes, "historicos": historicos}


if __name__ == "__main__":
    import os

    from dados import carregar_arff
    from validacao import validacao_cruzada

    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wine_quality.arff")
    atributos, alvo, _, _ = carregar_arff(caminho)
    folds = list(validacao_cruzada(alvo, 5, estratificada=False, semente=SEMENTE))
    experimento = avaliar_topologias_mlp(atributos, alvo, folds)
    for nome, resultado in experimento["resultados"].items():
        teste = resultado["teste"]
        print(
            f"{nome}: R²={teste['R²']['media']:.4f} EAM={teste['MAE']['media']:.4f} "
            f"treino={teste['Tempo Treino (s)']['media']:.3f}s teste={teste['Tempo Teste (s)']['media']:.4f}s"
        )
