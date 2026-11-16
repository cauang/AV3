"""Experimentos com o Perceptron Simples na classificação de crédito (AV3).

Varre taxa de aprendizado × número de épocas com a mesma validação cruzada
estratificada de 5 folds usada pelos demais modelos. A classe positiva é
"Mau pagador" (1). Métricas: matriz de confusão, acurácia, sensibilidade
(recall), especificidade, precisão e F1-Score, mais tempos de treino e teste.
"""
import os
import time

import numpy as np

import graficos
from dados import carregar_arff, padronizar
from metricas import matriz_confusao, metricas_binarias
from modelos import Perceptron
from validacao import validacao_cruzada

PASTA_RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEMENTE = 42
QUANTIDADE_FOLDS = 5
CLASSES = {0: "Bom pagador", 1: "Mau pagador"}
TAXAS = [0.001, 0.01, 0.1, 1.0]
EPOCAS = [1, 5, 10, 25, 50, 100, 200]
METRICAS = ["Acurácia", "Recall", "Especificidade", "Precisão", "F1-Score"]


def sem_padronizacao(atributos_treino, atributos_teste, colunas_numericas=None):
    """Mesma assinatura de `padronizar`, mas devolve os dados originais."""
    return atributos_treino.copy(), atributos_teste.copy()


def _media_desvio(valores):
    return {"media": float(np.mean(valores)), "desvio": float(np.std(valores, ddof=1))}


def avaliar_perceptron(atributos, alvo, folds, taxa, epocas, colunas_numericas=None,
                       normalizar=True, inicializacao="aleatoria", semente=SEMENTE):
    """Avalia uma configuração em todos os folds."""
    preparar = padronizar if normalizar else sem_padronizacao
    por_metrica = {}
    acuracia_treino, tempos_treino, tempos_teste, erros_finais = [], [], [], []
    previsto_total = np.zeros(len(alvo), dtype=int)
    historico = None

    for indices_treino, indices_teste in folds:
        treino, teste = preparar(atributos[indices_treino], atributos[indices_teste], colunas_numericas)
        alvo_treino = alvo[indices_treino].astype(int)
        alvo_teste = alvo[indices_teste].astype(int)

        modelo = Perceptron(taxa, epocas, inicializacao=inicializacao, semente=semente)
        inicio = time.perf_counter()
        modelo.fit(treino, alvo_treino)
        tempos_treino.append(time.perf_counter() - inicio)

        inicio = time.perf_counter()
        previsto = modelo.predict(teste)
        tempos_teste.append(time.perf_counter() - inicio)

        acuracia_treino.append(float(np.mean(modelo.predict(treino) == alvo_treino)))
        erros_finais.append(modelo.erros_por_epoca[-1] / len(alvo_treino))
        for metrica, valor in metricas_binarias(alvo_teste, previsto).items():
            por_metrica.setdefault(metrica, []).append(valor)
        previsto_total[indices_teste] = previsto
        if historico is None:
            historico = [e / len(alvo_treino) for e in modelo.erros_por_epoca]

    resumo = {m: _media_desvio(v) for m, v in por_metrica.items()}
    resumo["Acurácia treino"] = _media_desvio(acuracia_treino)
    resumo["Tempo Treino (s)"] = _media_desvio(tempos_treino)
    resumo["Tempo Teste (s)"] = _media_desvio(tempos_teste)
    return {
        "taxa": taxa, "epocas": epocas, "resumo": resumo,
        "matriz_confusao": matriz_confusao(alvo.astype(int), previsto_total, [0, 1]),
        "taxa_erro_ultima_epoca": _media_desvio(erros_finais),
        "historico_erros": historico,
    }


def varrer(atributos, alvo, colunas_numericas, normalizar=True, inicializacao="aleatoria"):
    folds = list(validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=True, semente=SEMENTE))
    resultados = []
    for taxa in TAXAS:
        for epocas in EPOCAS:
            resultados.append(avaliar_perceptron(
                atributos, alvo, folds, taxa, epocas, colunas_numericas, normalizar, inicializacao))
    return resultados


def _celula(resumo, metrica):
    return f"{resumo[metrica]['media']:.4f} ± {resumo[metrica]['desvio']:.4f}"


def imprimir_grade(resultados, titulo):
    print(f"\n{titulo}")
    print("| LR | Épocas | Acurácia | Recall (Sens.) | Especificidade | Precisão | F1-Score | Acur. treino | Treino (s) | Teste (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in resultados:
        s = r["resumo"]
        print(f"| {r['taxa']} | {r['epocas']} | " + " | ".join(
            _celula(s, m) for m in METRICAS + ["Acurácia treino"]) +
            f" | {s['Tempo Treino (s)']['media']:.4f} | {s['Tempo Teste (s)']['media']:.5f} |")


def salvar_csv(resultados, caminho):
    colunas = METRICAS + ["Acurácia treino", "Tempo Treino (s)", "Tempo Teste (s)"]
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("taxa_aprendizado,epocas," + ",".join(f"{m} media,{m} desvio" for m in colunas) + "\n")
        for r in resultados:
            valores = ",".join(f"{r['resumo'][m]['media']:.6f},{r['resumo'][m]['desvio']:.6f}" for m in colunas)
            arquivo.write(f"{r['taxa']},{r['epocas']},{valores}\n")


def executar(pasta_resultados=None, pasta_figuras=None):
    pasta_resultados = pasta_resultados or os.path.join(PASTA_RAIZ, "resultados")
    pasta_figuras = pasta_figuras or os.path.join(pasta_resultados, "figuras")
    os.makedirs(pasta_figuras, exist_ok=True)

    atributos, alvo, _, colunas_numericas = carregar_arff(os.path.join(PASTA_RAIZ, "classificacao_credito.arff"))
    print(f"[Perceptron | Crédito] {atributos.shape[0]} amostras, {atributos.shape[1]} colunas, "
          f"classe positiva = Mau pagador, {QUANTIDADE_FOLDS}-fold estratificado, Z-score no treino")

    resultados = varrer(atributos, alvo, colunas_numericas, normalizar=True)
    imprimir_grade(resultados, "Grade de hiperparâmetros (média ± desvio em 5 folds):")
    salvar_csv(resultados, os.path.join(pasta_resultados, "classificacao_perceptron.csv"))

    melhor = max(resultados, key=lambda r: r["resumo"]["F1-Score"]["media"])
    melhor_acuracia = max(resultados, key=lambda r: r["resumo"]["Acurácia"]["media"])
    print(f"\nMelhor F1: lr={melhor['taxa']}, épocas={melhor['epocas']} "
          f"(F1={melhor['resumo']['F1-Score']['media']:.4f}, "
          f"acurácia={melhor['resumo']['Acurácia']['media']:.4f})")
    print(f"Melhor acurácia: lr={melhor_acuracia['taxa']}, épocas={melhor_acuracia['epocas']} "
          f"({melhor_acuracia['resumo']['Acurácia']['media']:.4f})")

    matriz = melhor["matriz_confusao"]
    print(f"Matriz de confusão do melhor F1 (linhas = real, colunas = previsto; Bom, Mau): {matriz.tolist()}")

    # Com pesos iniciais zero, a taxa de aprendizado só reescala os pesos (previsões idênticas).
    folds = list(validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=True, semente=SEMENTE))
    zeros = [avaliar_perceptron(atributos, alvo, folds, taxa, 50, colunas_numericas, True, "zeros")
             for taxa in TAXAS]
    print("\nPesos iniciais = 0 (50 épocas): acurácia por taxa de aprendizado")
    for r in zeros:
        print(f"  lr={r['taxa']}: {_celula(r['resumo'], 'Acurácia')}")

    # ---------------- figuras
    for metrica, arquivo in (("Acurácia", "perceptron_acuracia.svg"), ("F1-Score", "perceptron_f1.svg")):
        series = {f"lr={taxa}": [next(r for r in resultados if r["taxa"] == taxa and r["epocas"] == e)
                                  ["resumo"][metrica]["media"] for e in EPOCAS] for taxa in TAXAS}
        graficos.linhas(os.path.join(pasta_figuras, arquivo),
                        f"Perceptron: {metrica} de teste vs épocas", EPOCAS, series, "Épocas", metrica)
    graficos.mapa_de_calor(
        os.path.join(pasta_figuras, "perceptron_confusao.svg"),
        f"Matriz de confusão: Perceptron (lr={melhor['taxa']}, {melhor['epocas']} épocas)",
        matriz.tolist(), list(CLASSES.values()))
    graficos.linhas(
        os.path.join(pasta_figuras, "perceptron_erros_por_epoca.svg"),
        "Perceptron: taxa de erro no treino por época (primeiro fold)",
        list(range(1, len(melhor["historico_erros"]) + 1)),
        {f"lr={melhor['taxa']}": melhor["historico_erros"]}, "Época", "Taxa de erro")

    def serial(r):
        return {"taxa": r["taxa"], "epocas": r["epocas"], "metricas": r["resumo"],
                "matriz_confusao": r["matriz_confusao"].tolist(),
                "taxa_erro_ultima_epoca": r["taxa_erro_ultima_epoca"]}

    return {
        "grade": [serial(r) for r in resultados],
        "melhor_f1": serial(melhor),
        "melhor_acuracia": serial(melhor_acuracia),
        "pesos_zero_50_epocas": [serial(r) for r in zeros],
    }


if __name__ == "__main__":
    executar()
