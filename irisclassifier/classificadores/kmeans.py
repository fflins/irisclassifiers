# classifiers/kmeans.py
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        """Inicializa os centroides escolhendo k pontos aleatórios do dataset."""
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(X.shape[0])
        self.centroids = X[random_indices[:self.k]]

    def _assign_clusters(self, X):
        """Atribui cada ponto de dado ao centroide mais próximo."""
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            # Calcula a distância euclidiana ao quadrado (mais rápido)
            distances[:, i] = np.sum((X - centroid)**2, axis=1)
        # Retorna o índice do centroide mais próximo para cada ponto
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        """Recalcula os centroides como a média dos pontos em cada cluster."""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            # Pega todos os pontos atribuídos ao cluster i
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        return new_centroids

    def fit(self, X):
        """Executa o algoritmo K-Means."""
        self._initialize_centroids(X)

        for _ in range(self.max_iters):
            self.labels_ = self._assign_clusters(X)
            
            old_centroids = np.copy(self.centroids)
            self.centroids = self._update_centroids(X)

            # Verifica a convergência (se os centroides pararam de mudar)
            if np.allclose(old_centroids, self.centroids):
                break
        
        return self