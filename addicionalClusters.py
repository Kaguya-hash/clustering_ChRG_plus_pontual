import numpy as np
from numpy.typing import NDArray
import math
from itertools import product
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering, estimate_bandwidth

MAX = math.inf

class ChRG_plus_pontual:

    def __init__(self, K, S, M, lambda_):

        if not self._is_natural(S) or not self._is_natural(K) or not self._is_natural(M):
            raise ValueError("S, K, and M must be natural numbers")

        if not isinstance(lambda_, (int, float)) or lambda_ < 0:
            raise ValueError("lambda must be a non-negative real number")

        self.S = int(S)
        self.K = int(K)
        self.M = int(M)
        self.lambda_ = float(lambda_)

        self.labels_ = None
        self.multi_geos_ = None

    @staticmethod
    def _is_natural(value):
        return isinstance(value, int) and value >= 1
    
    def L(self, multi_geo_1, multi_geo_2):
        K = len(multi_geo_1)
        K_ = len(multi_geo_2)

        res = MAX
        for k in range(K):
            for l in range(K_):
                aux = np.sum((multi_geo_1[k] - multi_geo_2[l]) ** 2)
                if aux < res:
                    res = aux

        return np.sqrt(res)

    def psi(self, geo, x):
        return np.sum((geo - x) ** 2)

    def rep_geo(self, L, D):
        return D[L].mean(axis=0)

    def xi_C(self, D, Q, geo):
        total = 0.0
        for k in range(len(Q)):
            for m in Q[k]:
                total += self.psi(D[m], geo[k])
        return total

    def alpha(self, Q, D, geo):
        for k in range(len(Q)):
            if Q[k]:
                geo[k] = self.rep_geo(Q[k], D)

    def beta_restrito(self, Q, D, geo, C):
        [ L.clear() for L in Q ]
        rep = {}
        for m in C:
            valores = np.array([self.psi(D[m], geo[k]) for k in range(len(geo))])
            minimo = valores.min()
            ks = np.where(np.isclose(valores, minimo))[0]
            if len(ks) == 1:
                Q[ks[0]].append(m)
            else:
                rep[m] = ks.tolist()
        
        listas_por_m = [[[m, k] for k in ks] for m, ks in rep.items()]
        combinacoes = list(product(*listas_por_m))
        combinacoes = [list(comb) for comb in combinacoes]

        m = MAX
        geo_aux = [None] * len(geo)
        for c in combinacoes:
            Q_aux = copy.deepcopy(Q)
            for p in c:
                Q_aux[p[1]].append(p[0])
            self.alpha(Q_aux, D, geo_aux)
            if self.xi_C(D, Q_aux, geo_aux) < m:
                m = self.xi_C(D, Q_aux, geo_aux)
                c_f = c
        
        for p in c_f:
            Q[p[1]].append(p[0])

    def LD_K_simples(self, C, D, K, geo_0, max = 8, tol = 0.1, debug=False, filename="imagem.png"):
        Q = [[] for _ in range(K)]
        diff = tol + 1
        ant = 0.0
        t = 0

        while diff > tol and t < max:
            self.beta_restrito(Q, D, geo_0, C)
            self.alpha(Q, D, geo_0)
            t = t + 1
            atua = self.xi_C(D, Q, geo_0)
            diff = abs(ant - atua)
            ant = atua


    def fit(self, D):
        N = len(D)
        K = self.K
        K_G = self.S
        tam_out = [int((N / K_aux) / self.M) for K_aux in range(1, N + 1)]
        beta = self.lambda_

        P = [[n] for n in range(N)]
        T = [[D[i].copy()] for i in range(N)]
        L_func = [None] * (N)
        K_ori = [None] * (N)

        for k in range(N):
            self.LD_K_simples(P[k], D, 1, T[k])

        min_d = [None] * N

        for k in range(N):
            m = MAX
            l = 0
            for l_aux in range(N):
                if l_aux != k:

                    if tam_out[N - 1] == 0:
                        m_aux = self.L(T[k], T[l_aux])
                    else:
                        s = min(len(P[k]), len(P[l_aux]))
                        if s <= tam_out[N - 1]:
                            m_aux = self.L(T[k], T[l_aux]) * (1 + beta * (tam_out[N - 1] - s))
                        else:
                            m_aux = self.L(T[k], T[l_aux])

                    if m_aux < m:
                        l = l_aux
                        m = m_aux
            min_d[k] = [l, m]

        min_v = MAX
        min_p = [0, 0]
        for k in range(N):
            if min_d[k][1] < min_v:
                min_v = min_d[k][1]
                min_p = [k, min_d[k][0]]

        if min_p[0] > min_p[1]: min_p = [min_p[1], min_p[0]]

        L_func[N - 1] = min_v

        for K_t in range(N, K, -1):

            sum = 0
            for i in range(K_t):
                if len(P[i]) > tam_out[K_t - 1]:
                    sum = sum + 1

            K_ori[K_t - 1] = sum

            if sum <= K:
                break

            k = min_p[0]
            l = min_p[1]

            P[k].extend(P[l])
            del P[l]
            T[k].extend(T[l])
            del T[l] 

            K_k = len(T[k])

            if K_k > K_G:
                Q  = [[] for k in range(K_k)]
                self.beta_restrito(Q, D, T[k], P[k])
                ind_k = sorted(range(K_k), key=lambda i: len(Q[i]), reverse=True)[:min(K_G, K_k)]
                T[k] = [T[k][i] for i in range(K_k) if i in ind_k]

            if K_k >= 2 * K_G:
                self.LD_K_simples(P[k], D, K_G, T[k])

            min_k = [None] * (K_t - 1)
            min_d[k][1] = MAX
            for k_aux in range(K_t - 1):

                if tam_out[K_t - 1] == 0:
                    min_k[k_aux] = self.L(T[k], T[k_aux])
                else:
                    s = min(len(P[k]), len(P[k_aux]))
                    if s <= tam_out[K_t - 1]:
                        min_k[k_aux] = self.L(T[k], T[k_aux]) * (1 + beta * (tam_out[K_t - 1] - s))
                    else: 
                        min_k[k_aux] = self.L(T[k], T[k_aux])

                if min_k[k_aux] < min_d[k][1] and k != k_aux:
                    min_d[k][1] = min_k[k_aux]
                    min_d[k][0] = k_aux

            for k_aux in range(K_t - 1):
                if k_aux == k:
                    continue

                if k_aux >= l:
                    min_d[k_aux] = min_d[k_aux + 1]

                if min_d[k_aux][0] == k or min_d[k_aux][0] == l:
                    min_d[k_aux][1] = MAX
                    for l_aux in range(K_t - 1):

                        if tam_out[K_t - 1] == 0:
                            m_aux = self.L(T[k_aux], T[l_aux])
                        else:
                            s = min(len(P[k_aux]), len(P[l_aux]))
                            if s <= tam_out[K_t - 1]:
                                m_aux = self.L(T[k_aux], T[l_aux]) * (1 + beta * (tam_out[K_t - 1] - s))
                            else:
                                m_aux = self.L(T[k_aux], T[l_aux])

                        if m_aux < min_d[k_aux][1] and k_aux != l_aux:
                            min_d[k_aux][1] = m_aux
                            min_d[k_aux][0] = l_aux
                else:
                    if min_d[k_aux][0] > l:
                        min_d[k_aux][0] = min_d[k_aux][0] - 1
                    
                    if min_k[k_aux] < min_d[k_aux][1]:
                        min_d[k_aux][1] = min_k[k_aux]
                        min_d[k_aux][0] = k

            min_v = MAX
            for k in range(K_t - 1):
                if min_d[k][1] < min_v:
                    min_v = min_d[k][1]
                    min_p = [k, min_d[k][0]]

            if min_p[0] > min_p[1]: min_p = [min_p[1], min_p[0]]

            L_func[(K_t - 1) - 1] = min_v
                
            target_index = (K_t - 1) - 1
        
        # initialize all points as outliers (-1)
        label_ = np.full(N, -1, dtype=int)

        # select the self.K largest clusters from partition P
        cluster_sizes = [(i, len(cluster)) for i, cluster in enumerate(P)]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [i for i, _ in cluster_sizes[:self.K]]

        # map original cluster index -> new compact label 0..(K_selected-1)
        mapping = {old: new for new, old in enumerate(top_k_indices)}

        for old_idx in top_k_indices:
            cluster = P[old_idx]
            for idx in cluster:
                if not isinstance(idx, int) or idx < 0 or idx >= N:
                    raise ValueError(f"Index {idx} out of valid range 0..{N-1}")
                if label_[idx] != -1:
                    raise ValueError(f"Index {idx} assigned to multiple clusters")
                label_[idx] = mapping[old_idx]

        # points remaining with label -1 are considered outliers
        self.labels_ = label_
        self.multi_geos_ = [T[i] for i in top_k_indices]

class AutoConnWardAgglomerative:
    
    def __init__(self, n_clusters=2, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        
        self.model_ = None
        self.labels_ = None

    def fit(self, X):
        connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=False)

        connectivity = 0.5 * (connectivity + connectivity.T)
        
        self.model_ = cluster.AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            linkage="ward", 
            connectivity=connectivity
        )
        
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        
        return self

class AutoConnAverageAgglomerative:
    """
    Wrapper for AgglomerativeClustering hardcoded to 'average' linkage 
    and 'cityblock' metric, with automatic connectivity calculation.
    """
    def __init__(self, n_clusters=2, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.model_ = None
        self.labels_ = None

    def fit(self, X):
        connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=False)

        connectivity = 0.5 * (connectivity + connectivity.T)
        
        self.model_ = cluster.AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="average",
            metric="cityblock",
            connectivity=connectivity
        )
        
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        return self


class AutoBandwidthMeanShift:

    def __init__(self, quantile=0.2):
        self.quantile = quantile
        self.model_ = None
        self.labels_ = None

    def fit(self, X):
        bandwidth = estimate_bandwidth(X, quantile=self.quantile)
    
        self.model_ = cluster.MeanShift(
            bandwidth=bandwidth, 
            bin_seeding=True
        )
    
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        return self