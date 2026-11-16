"""Análise do impacto da normalização Z-score (AV3, seção 3).

Para cada modelo, roda a MESMA validação cruzada (mesmos folds e sementes) com e
sem Z-score e compara as métricas. O Z-score usa média/desvio do treino de cada
fold e padroniza só as colunas numéricas (as colunas one-hot ficam como estão).
"""
import os
import time

import numpy as np

import experimento_perceptron as exp_perceptron
from dados import carregar_arff, padronizar
from metricas import metricas_binarias, metricas_regressao
from modelos import (BayesUnivariado, KNNClassificador, KNNRegressor, MLPRegressor,
                     Perceptron, RegressaoLinearMultipla)
from validacao import validacao_cruzada

PASTA_RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEMENTE = 42
QUANTIDADE_FOLDS = 5
K_PADRAO = 5
# Configuração de referência do Perceptron (fixa, para não escolher com base no Z-score).
PERCEPTRON_REFERENCIA = {"taxa": 0.01, "epocas": 50}
MLP_CONFIG = {"camadas_ocultas": (64, 32), "ativacao": "tanh", "taxa_aprendizado": 0.01, "epocas": 200}


def _avaliar(criar_modelo, atributos, alvo, folds, colunas_numericas, normalizar, tarefa):
    preparar = padronizar if normalizar else exp_perceptron.sem_padronizacao
    por_metrica, tempos = {}, []
    for indices_treino, indices_teste in folds:
        treino, teste = preparar(atributos[indices_treino], atributos[indices_teste], colunas_numericas)
        modelo = criar_modelo()
        inicio = time.perf_counter()
        with np.errstate(all="ignore"):
            modelo.fit(treino, alvo[indices_treino])
            previsto = modelo.predict(teste)
        tempos.append(time.perf_counter() - inicio)
        if tarefa == "classificacao":
            metricas = metricas_binarias(alvo[indices_teste].astype(int), np.asarray(previsto).astype(int))
        else:
            if not np.all(np.isfinite(previsto)):
                previsto = np.nan_to_num(previsto, nan=alvo.mean(), posinf=alvo.max(), neginf=alvo.min())
            metricas = metricas_regressao(alvo[indices_teste], previsto, atributos.shape[1])
        for nome, valor in metricas.items():
            por_metrica.setdefault(nome, []).append(float(valor))
    resumo = {n: {"media": float(np.mean(v)), "desvio": float(np.std(v, ddof=1))} for n, v in por_metrica.items()}
    resumo["Tempo Treino+Teste (s)"] = {"media": float(np.mean(tempos)), "desvio": float(np.std(tempos, ddof=1))}
    return resumo


def _comparar(modelos, atributos, alvo, folds, colunas_numericas, tarefa):
    return {
        nome: {
            "com": _avaliar(criar, atributos, alvo, folds, colunas_numericas, True, tarefa),
            "sem": _avaliar(criar, atributos, alvo, folds, colunas_numericas, False, tarefa),
        }
        for nome, criar in modelos.items()
    }


def _imprimir(resultado, metricas, titulo):
    print(f"\n{titulo}")
    print("| Modelo | Métrica | Sem Z-score | Com Z-score | Δ (com − sem) |")
    print("|---|---|---|---|---|")
    for nome, par in resultado.items():
        for metrica in metricas:
            sem, com = par["sem"][metrica]["media"], par["com"][metrica]["media"]
            print(f"| {nome} | {metrica} | {sem:.4f} ± {par['sem'][metrica]['desvio']:.4f} | "
                  f"{com:.4f} ± {par['com'][metrica]['desvio']:.4f} | {com - sem:+.4f} |")


def _salvar_csv(resultado, metricas, caminho):
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("modelo,metrica,sem_zscore_media,sem_zscore_desvio,com_zscore_media,com_zscore_desvio,delta\n")
        for nome, par in resultado.items():
            for metrica in metricas:
                sem, com = par["sem"][metrica], par["com"][metrica]
                arquivo.write(f"{nome},{metrica},{sem['media']:.6f},{sem['desvio']:.6f},"
                              f"{com['media']:.6f},{com['desvio']:.6f},{com['media'] - sem['media']:.6f}\n")


def executar(pasta_resultados=None):
    pasta_resultados = pasta_resultados or os.path.join(PASTA_RAIZ, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)

    # ------------------------------------------------------------ crédito
    atributos, alvo, _, numericas = carregar_arff(os.path.join(PASTA_RAIZ, "classificacao_credito.arff"))
    folds = list(validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=True, semente=SEMENTE))
    modelos = {
        "kNN (Euclidiana)": lambda: KNNClassificador(K_PADRAO, "euclidiana"),
        "kNN (Manhattan)": lambda: KNNClassificador(K_PADRAO, "manhattan"),
        "Bayesiano (Univariado)": lambda: BayesUnivariado(colunas_binarias=~numericas),
        f"Perceptron (lr={PERCEPTRON_REFERENCIA['taxa']}, {PERCEPTRON_REFERENCIA['epocas']} ép.)":
            lambda: Perceptron(PERCEPTRON_REFERENCIA["taxa"], PERCEPTRON_REFERENCIA["epocas"], semente=SEMENTE),
    }
    credito = _comparar(modelos, atributos, alvo, folds, numericas, "classificacao")
    metricas_c = ["Acurácia", "Recall", "Especificidade", "Precisão", "F1-Score"]
    _imprimir(credito, metricas_c, "[Z-score | Crédito] classe positiva = Mau pagador")
    _salvar_csv(credito, metricas_c, os.path.join(pasta_resultados, "zscore_credito.csv"))

    # Perceptron: grade inteira, para ver se a conclusão depende da configuração.
    com = exp_perceptron.varrer(atributos, alvo, numericas, normalizar=True)
    sem = exp_perceptron.varrer(atributos, alvo, numericas, normalizar=False)
    grade = {}
    for rotulo, resultados in (("com", com), ("sem", sem)):
        acuracias = [r["resumo"]["Acurácia"]["media"] for r in resultados]
        f1s = [r["resumo"]["F1-Score"]["media"] for r in resultados]
        grade[rotulo] = {
            "acuracia_media_da_grade": float(np.mean(acuracias)),
            "acuracia_melhor": float(np.max(acuracias)),
            "acuracia_pior": float(np.min(acuracias)),
            "f1_medio_da_grade": float(np.mean(f1s)),
            "f1_melhor": float(np.max(f1s)),
            "desvio_medio_entre_folds": float(np.mean([r["resumo"]["Acurácia"]["desvio"] for r in resultados])),
        }
    print("\n[Z-score | Crédito] Perceptron, grade completa (4 taxas × 7 épocas = 28 configurações)")
    print("| Critério | Sem Z-score | Com Z-score |")
    print("|---|---|---|")
    nomes = {"acuracia_media_da_grade": "Acurácia média da grade", "acuracia_melhor": "Melhor acurácia",
             "acuracia_pior": "Pior acurácia", "f1_medio_da_grade": "F1 médio da grade",
             "f1_melhor": "Melhor F1", "desvio_medio_entre_folds": "Desvio médio entre folds"}
    for chave, rotulo in nomes.items():
        print(f"| {rotulo} | {grade['sem'][chave]:.4f} | {grade['com'][chave]:.4f} |")

    # ------------------------------------------------------------ vinho
    atributos, alvo, _, _ = carregar_arff(os.path.join(PASTA_RAIZ, "wine_quality.arff"))
    folds = list(validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=False, semente=SEMENTE))
    modelos = {
        "kNN (Euclidiana)": lambda: KNNRegressor(K_PADRAO, "euclidiana"),
        "kNN (Manhattan)": lambda: KNNRegressor(K_PADRAO, "manhattan"),
        "Regressão Linear Múltipla": lambda: RegressaoLinearMultipla(),
        "MLP (64-32 Tanh)": lambda: MLPRegressor(tamanho_lote=64, semente=SEMENTE, **MLP_CONFIG),
    }
    vinho = _comparar(modelos, atributos, alvo, folds, None, "regressao")
    metricas_v = ["R²", "R² ajustado", "MSE", "RMSE", "MAE"]
    _imprimir(vinho, metricas_v, "[Z-score | Vinho]")
    _salvar_csv(vinho, metricas_v, os.path.join(pasta_resultados, "zscore_vinho.csv"))

    return {"credito": credito, "perceptron_grade_completa": grade, "vinho": vinho,
            "perceptron_referencia": PERCEPTRON_REFERENCIA}


if __name__ == "__main__":
    executar()
