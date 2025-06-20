# rbm.py
import numpy as np

class RBM:
    """
    Implementação de uma Máquina de Boltzmann Restrita (RBM)
    para aprendizado de características não supervisionado.
    """
    def __init__(self, num_visible, num_hidden):
        """
        Inicializa os parâmetros da RBM.

        Args:
            num_visible (int): Número de unidades na camada visível.
            num_hidden (int): Número de unidades na camada oculta.
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # Inicializa os pesos com valores aleatórios de uma distribuição normal
        # com desvio padrão pequeno.
        self.W = np.random.randn(num_visible, num_hidden) * 0.1
        
        # Inicializa os biases com zero.
        # 'b' é o bias da camada visível, 'c' é o bias da camada oculta.
        self.b = np.zeros((1, num_visible))
        self.c = np.zeros((1, num_hidden))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_prob(self, probs):
        """Amostra um estado binário (0 ou 1) a partir de um vetor de probabilidades."""
        return (probs > np.random.rand(*probs.shape)).astype(np.float32)

    def train(self, X, epochs=100, learning_rate=0.1, batch_size=10):
        """
        Treina a RBM usando Divergência Contrastiva (CD-1).
        """
        num_samples = X.shape[0]
        reconstruction_errors = []

        for epoch in range(epochs):
            # Embaralha os dados a cada época
            np.random.shuffle(X)
            
            epoch_error = 0.0
            for i in range(0, num_samples, batch_size):
                # Pega um mini-batch de dados
                v0 = X[i:i + batch_size]
                
                # --- Fase Positiva ---
                # Calcula as probabilidades da camada oculta e amostra h0
                prob_h0 = self._sigmoid(np.dot(v0, self.W) + self.c)
                h0 = self._sample_prob(prob_h0)

                # --- Fase Negativa (CD-1) ---
                # Reconstrói a camada visível a partir de h0
                # Para dados contínuos, usamos as probabilidades diretamente
                v1 = self._sigmoid(np.dot(h0, self.W.T) + self.b)
                # Calcula as novas probabilidades da camada oculta a partir de v1
                prob_h1 = self._sigmoid(np.dot(v1, self.W) + self.c)

                # --- Cálculo dos Gradientes ---
                # Associação positiva (dados reais)
                positive_grad = np.dot(v0.T, prob_h0)
                # Associação negativa (dados reconstruídos/fantasiados)
                negative_grad = np.dot(v1.T, prob_h1)

                # --- Atualização dos Pesos e Biases ---
                self.W += learning_rate * (positive_grad - negative_grad) / batch_size
                self.b += learning_rate * np.mean(v0 - v1, axis=0)
                self.c += learning_rate * np.mean(prob_h0 - prob_h1, axis=0)
                
                # Acumula o erro de reconstrução da época
                epoch_error += np.mean((v0 - v1)**2)
            
            # Armazena e imprime o erro da época
            avg_epoch_error = epoch_error / (num_samples / batch_size)
            reconstruction_errors.append(avg_epoch_error)
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs}, Erro de Reconstrução: {avg_epoch_error:.4f}")
        
        return reconstruction_errors
        
    def transform(self, X):
        """
        Usa a RBM treinada para extrair características dos dados.
        Isso é simplesmente a ativação da camada oculta.
        """
        # Retorna as probabilidades de ativação da camada oculta
        hidden_probs = self._sigmoid(np.dot(X, self.W) + self.c)
        return hidden_probs