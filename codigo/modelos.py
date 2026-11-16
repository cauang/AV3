import math
import random

import numpy as np


def calcular_distancias(amostras, base, tipo):
    """Matriz (len(amostras) x len(base)) de distâncias entre cada par de pontos."""
    if tipo == "euclidiana":
        quadrados = (
            (amostras ** 2).sum(axis=1)[:, None]
            + (base ** 2).sum(axis=1)[None, :]
            - 2 * amostras @ base.T
        )
        return np.sqrt(np.maximum(quadrados, 0.0))
    if tipo == "manhattan":
        return np.array([np.abs(base - amostra).sum(axis=1) for amostra in amostras])
    raise ValueError(f"distância desconhecida: {tipo}")


class _KNN:
    def __init__(self, k=5, distancia="euclidiana"):
        self.k = k
        self.distancia = distancia

    def _indices_dos_vizinhos(self, atributos, quantidade):
        distancias = calcular_distancias(atributos, self.atributos_treino, self.distancia)
        return np.argsort(distancias, axis=1)[:, :quantidade]

    def predict(self, atributos):
        vizinhos = self._indices_dos_vizinhos(atributos, self.k)
        return self._agregar(vizinhos, self.k)

    def prever_varios_k(self, atributos, lista_k):
        """Calcula as distâncias uma vez e devolve {k: previsões}."""
        vizinhos = self._indices_dos_vizinhos(atributos, max(lista_k))
        return {k: self._agregar(vizinhos, k) for k in lista_k}


class KNNClassificador(_KNN):
    def fit(self, atributos, alvo):
        self.atributos_treino = atributos
        self.alvo_treino = alvo.astype(int)
        self.classes = np.unique(self.alvo_treino)
        return self

    def _agregar(self, vizinhos, k):
        rotulos = self.alvo_treino[vizinhos[:, :k]]
        votos = np.stack([(rotulos == classe).sum(axis=1) for classe in self.classes], axis=1)
        return self.classes[votos.argmax(axis=1)]


class KNNRegressor(_KNN):
    def fit(self, atributos, alvo):
        self.atributos_treino = atributos
        self.alvo_treino = alvo
        return self

    def _agregar(self, vizinhos, k):
        return self.alvo_treino[vizinhos[:, :k]].mean(axis=1)


class BayesUnivariado:
    """Bayes ingênuo com um modelo univariado por atributo e por classe.

    Colunas contínuas usam uma gaussiana; colunas 0/1 (categorias codificadas) usam
    a frequência da categoria na classe, com suavização de Laplace.
    """

    def __init__(self, colunas_binarias=None, variancia_minima=1e-9):
        self.colunas_binarias = colunas_binarias
        self.variancia_minima = variancia_minima

    def fit(self, atributos, alvo):
        alvo = alvo.astype(int)
        if self.colunas_binarias is None:
            self.colunas_binarias = np.zeros(atributos.shape[1], dtype=bool)
        self.classes = np.unique(alvo)
        self.log_prioris = np.log([np.mean(alvo == classe) for classe in self.classes])

        por_classe = [atributos[alvo == classe] for classe in self.classes]
        self.medias = np.array([amostras.mean(axis=0) for amostras in por_classe])
        self.variancias = np.array([amostras.var(axis=0) for amostras in por_classe]) + self.variancia_minima
        self.frequencias = np.array([
            (amostras.sum(axis=0) + 1) / (len(amostras) + 2) for amostras in por_classe
        ])
        return self

    def predict(self, atributos):
        continuas = ~self.colunas_binarias
        log_posterioris = []
        for i in range(len(self.classes)):
            log_gaussiana = -0.5 * (
                np.log(2 * np.pi * self.variancias[i, continuas])
                + (atributos[:, continuas] - self.medias[i, continuas]) ** 2 / self.variancias[i, continuas]
            )
            frequencia = self.frequencias[i, self.colunas_binarias]
            log_categorias = (
                atributos[:, self.colunas_binarias] * np.log(frequencia)
                + (1 - atributos[:, self.colunas_binarias]) * np.log(1 - frequencia)
            )
            log_posterioris.append(
                self.log_prioris[i] + log_gaussiana.sum(axis=1) + log_categorias.sum(axis=1)
            )
        return self.classes[np.argmax(np.stack(log_posterioris, axis=1), axis=1)]


class RegressaoLinearMultipla:
    """Mínimos quadrados pelas equações normais, com pseudo-inversa por estabilidade."""

    @staticmethod
    def _com_intercepto(atributos):
        return np.hstack([np.ones((len(atributos), 1)), atributos])

    def fit(self, atributos, alvo):
        matriz = self._com_intercepto(atributos)
        self.coeficientes = np.linalg.pinv(matriz.T @ matriz) @ matriz.T @ alvo
        return self

    def predict(self, atributos):
        return self._com_intercepto(atributos) @ self.coeficientes


class MLPRegressor:
    """Perceptron de Múltiplas Camadas para regressão, implementado à mão.

    Treina por retropropagação do erro com descida de gradiente em mini-lotes.
    A saída é linear (uma única nota prevista) e o alvo é padronizado
    internamente durante o treino, voltando à escala original em `predict`.
    """

    def __init__(self, camadas_ocultas=(64,), ativacao="relu", taxa_aprendizado=0.01,
                 epocas=200, tamanho_lote=64, semente=42, normalizar_alvo=True,
                 momento=0.0, l2=0.0):
        self.camadas_ocultas = tuple(camadas_ocultas)
        self.ativacao = ativacao
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.semente = semente
        self.normalizar_alvo = normalizar_alvo
        self.momento = momento
        self.l2 = l2
        self.historico_perda = []

    # ------------------------------------------------------------------ ativações
    def _aplicar_ativacao(self, z):
        if self.ativacao == "relu":
            return np.maximum(z, 0.0)
        if self.ativacao == "tanh":
            return np.tanh(z)
        if self.ativacao == "sigmoid":
            # forma estável de sigmoide: 0,5 * (1 + tanh(z/2))
            return 0.5 * (1.0 + np.tanh(0.5 * z))
        if self.ativacao == "linear":
            return z
        raise ValueError(f"ativação desconhecida: {self.ativacao}")

    def _derivada_ativacao(self, z, ativacao):
        if self.ativacao == "relu":
            return (z > 0).astype(float)
        if self.ativacao == "tanh":
            return 1.0 - ativacao ** 2
        if self.ativacao == "sigmoid":
            return ativacao * (1.0 - ativacao)
        if self.ativacao == "linear":
            return np.ones_like(z)
        raise ValueError(f"ativação desconhecida: {self.ativacao}")

    # ------------------------------------------------------------------ inicialização
    def _inicializar_pesos(self, quantidade_entradas):
        sorteio = random.Random(self.semente)
        tamanhos = [quantidade_entradas] + list(self.camadas_ocultas) + [1]
        pesos, vieses = [], []
        for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]):
            if self.ativacao == "relu":
                desvio = math.sqrt(2.0 / entrada)  # inicialização de He
            else:
                desvio = math.sqrt(2.0 / (entrada + saida))  # inicialização de Xavier
            pesos.append(np.array([
                [sorteio.gauss(0.0, desvio) for _ in range(saida)]
                for _ in range(entrada)
            ]))
            vieses.append(np.zeros(saida))
        return pesos, vieses

    # ------------------------------------------------------------------ propagação
    def _propagar(self, atributos):
        ativacoes = [atributos]
        pre_ativacoes = []
        atual = atributos
        ultima_camada = len(self.pesos) - 1
        for indice, (pesos, vieses) in enumerate(zip(self.pesos, self.vieses)):
            z = atual @ pesos + vieses
            pre_ativacoes.append(z)
            atual = z if indice == ultima_camada else self._aplicar_ativacao(z)
            ativacoes.append(atual)
        return ativacoes, pre_ativacoes

    # ------------------------------------------------------------------ treino
    def fit(self, atributos, alvo):
        atributos = np.asarray(atributos, dtype=float)
        alvo = np.asarray(alvo, dtype=float)
        self.pesos, self.vieses = self._inicializar_pesos(atributos.shape[1])

        if self.normalizar_alvo:
            self.media_alvo = float(alvo.mean())
            self.desvio_alvo = float(alvo.std()) or 1.0
        else:
            self.media_alvo, self.desvio_alvo = 0.0, 1.0
        alvo_normalizado = (alvo - self.media_alvo) / self.desvio_alvo

        quantidade = len(atributos)
        sorteio = random.Random(self.semente + 1)
        indices = list(range(quantidade))
        velocidades_pesos = [np.zeros_like(pesos) for pesos in self.pesos]
        velocidades_vieses = [np.zeros_like(vieses) for vieses in self.vieses]
        self.historico_perda = []

        for _ in range(self.epocas):
            sorteio.shuffle(indices)
            erro_total = 0.0
            for inicio in range(0, quantidade, self.tamanho_lote):
                lote = indices[inicio:inicio + self.tamanho_lote]
                atributos_lote = atributos[lote]
                alvo_lote = alvo_normalizado[lote]

                ativacoes, pre_ativacoes = self._propagar(atributos_lote)
                erro = ativacoes[-1].ravel() - alvo_lote
                erro_total += float(np.dot(erro, erro))

                # retropropagação
                delta = (2.0 / len(lote)) * erro.reshape(-1, 1)
                gradientes_pesos = [None] * len(self.pesos)
                gradientes_vieses = [None] * len(self.pesos)
                for indice in range(len(self.pesos) - 1, -1, -1):
                    gradientes_pesos[indice] = ativacoes[indice].T @ delta
                    gradientes_vieses[indice] = delta.sum(axis=0)
                    if indice > 0:
                        delta = (delta @ self.pesos[indice].T) * self._derivada_ativacao(
                            pre_ativacoes[indice - 1], ativacoes[indice]
                        )

                # atualização dos parâmetros
                for indice in range(len(self.pesos)):
                    if self.l2:
                        gradientes_pesos[indice] = gradientes_pesos[indice] + self.l2 * self.pesos[indice]
                    velocidades_pesos[indice] = (
                        self.momento * velocidades_pesos[indice]
                        - self.taxa_aprendizado * gradientes_pesos[indice]
                    )
                    velocidades_vieses[indice] = (
                        self.momento * velocidades_vieses[indice]
                        - self.taxa_aprendizado * gradientes_vieses[indice]
                    )
                    self.pesos[indice] += velocidades_pesos[indice]
                    self.vieses[indice] += velocidades_vieses[indice]

            self.historico_perda.append((erro_total / quantidade) * (self.desvio_alvo ** 2))

        return self

    # ------------------------------------------------------------------ previsão
    def predict(self, atributos):
        atributos = np.asarray(atributos, dtype=float)
        ativacoes, _ = self._propagar(atributos)
        return ativacoes[-1].ravel() * self.desvio_alvo + self.media_alvo

    def quantidade_de_parametros(self, quantidade_entradas):
        tamanhos = [quantidade_entradas] + list(self.camadas_ocultas) + [1]
        return sum(entrada * saida + saida for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]))


class Perceptron:
    """Perceptron Simples (Rosenblatt) para classificação binária, implementado à mão.

    Regra de aprendizado online: para cada amostra, saída = degrau(w·x + b) e, se
    errou, w += taxa * (y - saída) * x  e  b += taxa * (y - saída).
    As amostras são embaralhadas a cada época (semente fixa, reprodutível).

    Com `inicializacao="zeros"` (versão clássica) a taxa de aprendizado só muda a
    escala dos pesos e não altera as previsões; por isso o padrão é uma
    inicialização aleatória pequena, que torna a taxa relevante no experimento.
    """

    def __init__(self, taxa_aprendizado=0.01, epocas=100, inicializacao="aleatoria",
                 semente=42, embaralhar=True):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.inicializacao = inicializacao
        self.semente = semente
        self.embaralhar = embaralhar
        self.erros_por_epoca = []

    @staticmethod
    def _degrau(valor):
        return np.where(valor >= 0, 1, 0)

    def fit(self, atributos, alvo):
        atributos = np.asarray(atributos, dtype=float)
        alvo = np.asarray(alvo).astype(int)
        quantidade, quantidade_atributos = atributos.shape
        sorteio = random.Random(self.semente)

        if self.inicializacao == "zeros":
            self.pesos = np.zeros(quantidade_atributos)
            self.vies = 0.0
        else:
            self.pesos = np.array([sorteio.gauss(0.0, 0.01) for _ in range(quantidade_atributos)])
            self.vies = sorteio.gauss(0.0, 0.01)

        indices = list(range(quantidade))
        self.erros_por_epoca = []
        for _ in range(self.epocas):
            if self.embaralhar:
                sorteio.shuffle(indices)
            erros = 0
            for i in indices:
                x_i = atributos[i]
                previsto = 1 if np.dot(x_i, self.pesos) + self.vies >= 0 else 0
                if previsto != alvo[i]:
                    ajuste = self.taxa_aprendizado * (alvo[i] - previsto)
                    self.pesos += ajuste * x_i
                    self.vies += ajuste
                    erros += 1
            self.erros_por_epoca.append(erros)
        return self

    def predict(self, atributos):
        return self._degrau(np.asarray(atributos, dtype=float) @ self.pesos + self.vies)
