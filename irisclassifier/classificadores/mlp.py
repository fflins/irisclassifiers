# mlp.py
import numpy as np

class MLP:
    """
    Uma implementação de um Perceptron de Múltiplas Camadas (MLP) do zero
    com uma camada oculta para classificação.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa os pesos e biases da rede neural.

        Args:
            input_size (int): Número de neurônios na camada de entrada (features).
            hidden_size (int): Número de neurônios na camada oculta.
            output_size (int): Número de neurônios na camada de saída (classes).
        """
        # Inicializa os pesos com valores aleatórios pequenos para evitar saturação da sigmoide
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        print("Rede MLP inicializada.")
        print(f"Pesos W1 (Entrada -> Oculta): {self.W1.shape}")
        print(f"Pesos W2 (Oculta -> Saída): {self.W2.shape}")

    def _sigmoid(self, z):
        """Função de ativação Sigmoide."""
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        """Derivada da função Sigmoide."""
        s = self._sigmoid(z)
        return s * (1 - s)

    def _softmax(self, z):
        """Função de ativação Softmax para a camada de saída."""
        # Subtrair o máximo previne instabilidade numérica (overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Executa o forward pass (propagação direta) através da rede.
        """
        # Camada Oculta
        # z1 é o output linear; a1 é o output após a ativação
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)

        # Camada de Saída
        # z2 é o output linear; y_pred é a probabilidade final após o softmax
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_pred = self._softmax(self.z2)
        
        return y_pred

    def _compute_loss(self, y_true, y_pred):
        """
        Calcula a perda da Entropia Cruzada Categórica.
        
        Args:
            y_true (np.array): Rótulos verdadeiros em formato one-hot.
            y_pred (np.array): Predições do modelo (saída do softmax).
        """
        num_samples = len(y_true)
        # Adicionar um valor pequeno (epsilon) para evitar log(0)
        epsilon = 1e-9
        loss = - (1 / num_samples) * np.sum(y_true * np.log(y_pred + epsilon))
        return loss

    def backward(self, X, y_true, y_pred):
        """
        Executa o backward pass (backpropagation) para calcular os gradientes.
        """
        num_samples = X.shape[0]

        # 1. Gradiente do erro na camada de saída
        # O gradiente da entropia cruzada com softmax é simplesmente (predição - real)
        dZ2 = y_pred - y_true

        # 2. Gradientes para os pesos e bias da camada de saída (W2, b2)
        dW2 = (1 / num_samples) * np.dot(self.a1.T, dZ2)
        db2 = (1 / num_samples) * np.sum(dZ2, axis=0, keepdims=True)

        # 3. Propagar o erro para a camada oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._sigmoid_derivative(self.z1)

        # 4. Gradientes para os pesos e bias da camada oculta (W1, b1)
        dW1 = (1 / num_samples) * np.dot(X.T, dZ1)
        db1 = (1 / num_samples) * np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Treina a rede neural usando gradiente descendente.
        """
        loss_history = []

        for epoch in range(epochs):
            # 1. Forward Pass
            y_pred = self.forward(X_train)

            # 2. Calcular a perda
            loss = self._compute_loss(y_train, y_pred)
            loss_history.append(loss)

            # 3. Backward Pass (calcular gradientes)
            dW1, db1, dW2, db2 = self.backward(X_train, y_train, y_pred)

            # 4. Atualizar os pesos e biases (Gradiente Descendente)
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            # Imprimir o progresso do treinamento
            if (epoch + 1) % 1000 == 0:
                print(f"Época {epoch + 1}/{epochs}, Perda (Loss): {loss:.4f}")
        
        return loss_history