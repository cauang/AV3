import numpy as np

# Rótulos em pt-BR usados apenas na exibição (tabelas e gráficos).
# As chaves internas, os CSVs e o JSON continuam com os nomes MSE, RMSE e MAE.
ROTULOS_EXIBICAO = {
    "MSE": "EQM",
    "RMSE": "REQM",
    "MAE": "EAM",
}


def rotular(metrica):
    """Nome da métrica para exibição em português (ou o próprio nome, se não houver)."""
    return ROTULOS_EXIBICAO.get(metrica, metrica)


def matriz_confusao(alvo_real, alvo_previsto, classes):
    posicao = {classe: i for i, classe in enumerate(classes)}
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for real, previsto in zip(alvo_real, alvo_previsto):
        matriz[posicao[real], posicao[previsto]] += 1
    return matriz


def precisao_recall_f1_por_classe(matriz):
    """Devolve três vetores (um valor por classe) a partir da matriz de confusão."""
    acertos = np.diag(matriz).astype(float)
    total_previsto = matriz.sum(axis=0)
    total_real = matriz.sum(axis=1)

    precisao = np.divide(acertos, total_previsto, out=np.zeros_like(acertos), where=total_previsto > 0)
    recall = np.divide(acertos, total_real, out=np.zeros_like(acertos), where=total_real > 0)
    soma = precisao + recall
    f1 = np.divide(2 * precisao * recall, soma, out=np.zeros_like(acertos), where=soma > 0)
    return precisao, recall, f1


def metricas_classificacao(alvo_real, alvo_previsto, classes):
    """Acurácia e média macro de precisão, recall e F1."""
    matriz = matriz_confusao(alvo_real, alvo_previsto, classes)
    precisao, recall, f1 = precisao_recall_f1_por_classe(matriz)
    return {
        "Acurácia": np.trace(matriz) / matriz.sum(),
        "Precisão": precisao.mean(),
        "Recall": recall.mean(),
        "F1-Score": f1.mean(),
    }


def metricas_binarias(alvo_real, alvo_previsto, classe_positiva=1):
    """Métricas da classe positiva num problema de duas classes (0 e 1).

    Sensibilidade (recall) = VP/(VP+FN); Especificidade = VN/(VN+FP);
    Precisão = VP/(VP+FP); F1 = média harmônica de precisão e recall.
    """
    matriz = matriz_confusao(alvo_real, alvo_previsto, [0, 1])
    if classe_positiva == 1:
        vn, fp, fn, vp = matriz[0, 0], matriz[0, 1], matriz[1, 0], matriz[1, 1]
    else:
        vp, fn, fp, vn = matriz[0, 0], matriz[0, 1], matriz[1, 0], matriz[1, 1]

    def razao(numerador, denominador):
        return float(numerador) / denominador if denominador > 0 else 0.0

    precisao = razao(vp, vp + fp)
    recall = razao(vp, vp + fn)
    return {
        "Acurácia": razao(vp + vn, matriz.sum()),
        "Recall": recall,
        "Especificidade": razao(vn, vn + fp),
        "Precisão": precisao,
        "F1-Score": razao(2 * precisao * recall, precisao + recall),
    }


def metricas_regressao(alvo_real, alvo_previsto, quantidade_atributos):
    n = len(alvo_real)
    erro = alvo_real - alvo_previsto
    soma_quadrados_residuos = np.sum(erro ** 2)
    soma_quadrados_total = np.sum((alvo_real - alvo_real.mean()) ** 2)

    mse = soma_quadrados_residuos / n
    r2 = 1 - soma_quadrados_residuos / soma_quadrados_total
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - quantidade_atributos - 1)
    return {
        "R²": r2,
        "R² ajustado": r2_ajustado,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": np.mean(np.abs(erro)),
    }


def taxa_de_acerto_com_tolerancia(alvo_real, alvo_previsto, tolerancia):
    return float(np.mean(np.abs(alvo_real - alvo_previsto) <= tolerancia))
