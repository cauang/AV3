"""Verificação independente da implementação do MLP.

Confere, sem bibliotecas de aprendizado de máquina:
- formato das matrizes de peso (arquitetura realmente multicamadas);
- forward calculado à mão vs `predict`;
- gradiente analítico vs diferenças centrais (backpropagation);
- necessidade da camada oculta (XOR) e aproximação não linear;
- escala do alvo, determinismo da semente e opções de momento/L2;
- consistência das métricas com o pipeline real do vinho (1 fold).

Uso: `python verificacao_mlp.py`
"""
import os

import numpy as np

from dados import carregar_arff, padronizar
from metricas import metricas_regressao
from modelos import MLPRegressor, RegressaoLinearMultipla
from validacao import validacao_cruzada

falhas = []


def checar(nome, condicao, detalhe=""):
    print(("  OK   " if condicao else "  FALHA") + f" | {nome}" + (f" | {detalhe}" if detalhe else ""))
    if not condicao:
        falhas.append(nome)


def perda(modelo, X, y):
    ativacoes, _ = modelo._propagar(X)
    erro = ativacoes[-1].ravel() - y
    return float(np.mean(erro ** 2))


def gradientes_analiticos(modelo, X, y):
    ativacoes, pre = modelo._propagar(X)
    erro = ativacoes[-1].ravel() - y
    delta = (2.0 / len(X)) * erro.reshape(-1, 1)
    gW = [None] * len(modelo.pesos)
    gb = [None] * len(modelo.pesos)
    for i in range(len(modelo.pesos) - 1, -1, -1):
        gW[i] = ativacoes[i].T @ delta
        gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ modelo.pesos[i].T) * modelo._derivada_ativacao(
                pre[i - 1], ativacoes[i]
            )
    return gW, gb


def erro_de_gradiente(camadas, ativacao, entradas=5, amostras=7, semente=0):
    rng = np.random.default_rng(semente)
    X = rng.normal(size=(amostras, entradas))
    y = rng.normal(size=amostras)
    modelo = MLPRegressor(
        camadas_ocultas=camadas, ativacao=ativacao, normalizar_alvo=False, semente=3
    )
    modelo.pesos, modelo.vieses = modelo._inicializar_pesos(entradas)
    gW, gb = gradientes_analiticos(modelo, X, y)

    eps = 1e-6
    pior = 0.0
    for i in range(len(modelo.pesos)):
        for indice in np.ndindex(modelo.pesos[i].shape):
            original = modelo.pesos[i][indice]
            modelo.pesos[i][indice] = original + eps
            mais = perda(modelo, X, y)
            modelo.pesos[i][indice] = original - eps
            menos = perda(modelo, X, y)
            modelo.pesos[i][indice] = original
            numerico = (mais - menos) / (2 * eps)
            escala = max(1.0, abs(numerico), abs(gW[i][indice]))
            pior = max(pior, abs(numerico - gW[i][indice]) / escala)
    for i in range(len(modelo.vieses)):
        for indice in np.ndindex(modelo.vieses[i].shape):
            original = modelo.vieses[i][indice]
            modelo.vieses[i][indice] = original + eps
            mais = perda(modelo, X, y)
            modelo.vieses[i][indice] = original - eps
            menos = perda(modelo, X, y)
            modelo.vieses[i][indice] = original
            numerico = (mais - menos) / (2 * eps)
            escala = max(1.0, abs(numerico), abs(gb[i][indice]))
            pior = max(pior, abs(numerico - gb[i][indice]) / escala)
    return pior


print("1) Arquitetura realmente multicamadas")
modelo = MLPRegressor(camadas_ocultas=(128, 64, 32), ativacao="tanh", semente=1)
modelo.pesos, modelo.vieses = modelo._inicializar_pesos(11)
formatos_W = [p.shape for p in modelo.pesos]
formatos_b = [b.shape for b in modelo.vieses]
checar("4 matrizes de pesos (3 ocultas + saída)", formatos_W == [(11, 128), (128, 64), (64, 32), (32, 1)], str(formatos_W))
checar("4 vetores de vieses", formatos_b == [(128,), (64,), (32,), (1,)], str(formatos_b))
checar("contagem de parâmetros bate com a soma", modelo.quantidade_de_parametros(11) == sum(p.size + b.size for p, b in zip(modelo.pesos, modelo.vieses)),
       f"{modelo.quantidade_de_parametros(11)}")

print("\n2) Forward manual == predict (Tanh, 3 camadas)")
rng = np.random.default_rng(5)
X = rng.normal(size=(4, 11))
y = rng.normal(size=4)
modelo = MLPRegressor(camadas_ocultas=(8, 6, 4), ativacao="tanh", semente=9)
modelo.fit(X, y)
a = X
for i, (W, b) in enumerate(zip(modelo.pesos, modelo.vieses)):
    a = a @ W + b
    if i < len(modelo.pesos) - 1:
        a = np.tanh(a)
previsto_manual = a.ravel() * modelo.desvio_alvo + modelo.media_alvo
checar("predict reproduz o forward calculado à mão", np.allclose(previsto_manual, modelo.predict(X)), f"erro={np.max(np.abs(previsto_manual - modelo.predict(X))):.2e}")

print("\n3) Gradiente analítico vs numérico")
for camadas in [(4,), (8, 4), (8, 6, 4)]:
    for ativacao in ("tanh", "sigmoid", "relu"):
        pior = erro_de_gradiente(camadas, ativacao)
        checar(f"camadas={camadas} ativação={ativacao}", pior < 1e-6, f"erro relativo máx={pior:.2e}")

print("\n4) A camada oculta é necessária (XOR)")
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_xor = np.array([0.0, 1.0, 1.0, 0.0])
linear = MLPRegressor(camadas_ocultas=(), ativacao="linear", taxa_aprendizado=0.1,
                      epocas=4000, tamanho_lote=4, normalizar_alvo=False, semente=0).fit(X_xor, y_xor)
oculta = MLPRegressor(camadas_ocultas=(8,), ativacao="tanh", taxa_aprendizado=0.1,
                      epocas=4000, tamanho_lote=4, normalizar_alvo=False, semente=0).fit(X_xor, y_xor)
mse_linear = perda(linear, X_xor, y_xor)
mse_oculta = perda(oculta, X_xor, y_xor)
checar("modelo sem camada oculta não resolve XOR", mse_linear > 0.2, f"EQM={mse_linear:.4f}")
checar("MLP com camada oculta resolve XOR", mse_oculta < 0.02, f"EQM={mse_oculta:.4f}")

print("\n5) Aproximação de função não linear (seno + quadrado)")
rng = np.random.default_rng(1)
X = rng.uniform(-2, 2, size=(400, 2))
y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
Xtr, Xte, ytr, yte = X[:300], X[300:], y[:300], y[300:]
linear = RegressaoLinearMultipla().fit(Xtr, ytr)
mlp = MLPRegressor(camadas_ocultas=(32, 16), ativacao="tanh", taxa_aprendizado=0.01,
                   epocas=400, tamanho_lote=32, semente=42).fit(Xtr, ytr)
r2_linear = metricas_regressao(yte, linear.predict(Xte), 2)["R²"]
r2_mlp = metricas_regressao(yte, mlp.predict(Xte), 2)["R²"]
checar("MLP supera claramente o modelo linear em função não linear", r2_mlp > r2_linear + 0.05,
       f"R² linear={r2_linear:.4f} | R² MLP={r2_mlp:.4f}")

print("\n6) Escala do alvo preservada")
X = np.linspace(-2, 2, 200).reshape(-1, 1)
y = 7.5 + 2.0 * X.ravel()
modelo = MLPRegressor(camadas_ocultas=(16,), ativacao="tanh", taxa_aprendizado=0.01,
                      epocas=600, tamanho_lote=32, semente=0).fit(X, y)
previsto = modelo.predict(X)
checar("previsões na escala original (média próxima de 7,5)", abs(previsto.mean() - y.mean()) < 0.1,
       f"média prevista={previsto.mean():.3f} vs real={y.mean():.3f}")
checar("tendência aprendida", np.corrcoef(previsto, y)[0, 1] > 0.99, f"correlação={np.corrcoef(previsto, y)[0, 1]:.5f}")

print("\n7) Determinismo e semente")
X = rng.normal(size=(200, 3))
y = rng.normal(size=200)
modelo_a = MLPRegressor(camadas_ocultas=(16,), ativacao="tanh", epocas=30, semente=42).fit(X, y)
modelo_b = MLPRegressor(camadas_ocultas=(16,), ativacao="tanh", epocas=30, semente=42).fit(X, y)
modelo_c = MLPRegressor(camadas_ocultas=(16,), ativacao="tanh", epocas=30, semente=7).fit(X, y)
checar("mesma semente gera o mesmo modelo", np.allclose(modelo_a.predict(X), modelo_b.predict(X)))
checar("semente diferente gera modelo diferente", not np.allclose(modelo_a.predict(X), modelo_c.predict(X)))

print("\n8) Curva de perda e treino")
rng8 = np.random.default_rng(11)
X8 = rng8.normal(size=(300, 3))
y8 = np.sin(X8[:, 0]) + 0.5 * X8[:, 1] ** 2 - X8[:, 2]
modelo8 = MLPRegressor(camadas_ocultas=(16,), ativacao="tanh", epocas=60, semente=42).fit(X8, y8)
historico = np.array(modelo8.historico_perda)
checar("histórico tem uma entrada por época", len(historico) == 60, str(len(historico)))
checar("perda cai do início para o fim (dados aprendíveis)", historico[-1] < historico[0] * 0.7,
       f"início={historico[0]:.4f} fim={historico[-1]:.4f}")
perda_final_real = float(np.mean((modelo8.predict(X8) - y8) ** 2))
print(f"         | EQM final recalculado com predict após o último update: {perda_final_real:.4f} "
      f"(último ponto do histórico: {historico[-1]:.4f})")

print("\n9) Momento e L2 opcionais")
base = dict(camadas_ocultas=(16,), ativacao="tanh", epocas=50, semente=42)
sem = MLPRegressor(**base).fit(X, y)
com = MLPRegressor(**base, momento=0.9).fit(X, y)
checar("momento altera o resultado do treino", not np.allclose(sem.predict(X), com.predict(X)))
norma_sem = sum(float(np.sum(W ** 2)) for W in sem.pesos)
norma_com = sum(float(np.sum(W ** 2)) for W in MLPRegressor(**base, l2=0.1).fit(X, y).pesos)
checar("L2 reduz a norma dos pesos", norma_com < norma_sem, f"sem={norma_sem:.3f} com={norma_com:.3f}")

print("\n10) Pipeline completo (1 fold do vinho, topologia Tanh 64-32)")
caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wine_quality.arff")
atributos, alvo, _, _ = carregar_arff(caminho)
indices_treino, indices_teste = next(iter(validacao_cruzada(alvo, 5, estratificada=False, semente=42)))
treino, teste = padronizar(atributos[indices_treino], atributos[indices_teste])
alvo_treino, alvo_teste = alvo[indices_treino], alvo[indices_teste]
modelo = MLPRegressor(camadas_ocultas=(64, 32), ativacao="tanh", taxa_aprendizado=0.01,
                      epocas=200, tamanho_lote=64, semente=42).fit(treino, alvo_treino)
previsto = modelo.predict(teste)
residuo = alvo_teste - previsto
r2_manual = 1 - np.sum(residuo ** 2) / np.sum((alvo_teste - alvo_teste.mean()) ** 2)
r2_biblioteca = metricas_regressao(alvo_teste, previsto, 11)["R²"]
checar("R² recalculado à mão confere com metricas_regressao", abs(r2_manual - r2_biblioteca) < 1e-12,
       f"manual={r2_manual:.6f} função={r2_biblioteca:.6f}")
checar("R² de teste plausível (0,2 a 0,5)", 0.2 < r2_biblioteca < 0.5, f"R²={r2_biblioteca:.4f}")
checar("previsões finitas e no intervalo razoável", np.all(np.isfinite(previsto)) and previsto.min() > 0 and previsto.max() < 12,
       f"min={previsto.min():.2f} max={previsto.max():.2f}")

print("\nResumo:", "todas as verificações passaram" if not falhas else f"{len(falhas)} falha(s): {falhas}")
