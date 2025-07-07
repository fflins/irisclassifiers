# rbm.py
import numpy as np

import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        self.W = np.random.normal(0, 0.01, (num_visible, num_hidden))
        self.b = np.zeros((1, num_visible))
        self.c = np.zeros((1, num_hidden))
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _sample_prob(self, probs):
        return (probs > np.random.rand(*probs.shape)).astype(np.float32)
    
    def train(self, X, epochs=100, learning_rate=0.001, batch_size=10):
        num_samples = X.shape[0]
        reconstruction_errors = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            epoch_error = 0.0
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                v0 = X_shuffled[i:batch_end]
                actual_batch_size = v0.shape[0]
                
                prob_h0 = self._sigmoid(np.dot(v0, self.W) + self.c)
                h0 = self._sample_prob(prob_h0)
                
                v1 = np.dot(h0, self.W.T) + self.b
                prob_h1 = self._sigmoid(np.dot(v1, self.W) + self.c)
                
                positive_grad = np.dot(v0.T, prob_h0)
                negative_grad = np.dot(v1.T, prob_h1)
                
                self.W += learning_rate * (positive_grad - negative_grad) / actual_batch_size
                self.b += learning_rate * np.mean(v0 - v1, axis=0, keepdims=True)
                self.c += learning_rate * np.mean(prob_h0 - prob_h1, axis=0, keepdims=True)
                
                epoch_error += np.mean((v0 - v1)**2)
            
            avg_epoch_error = epoch_error / (num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0))
            reconstruction_errors.append(avg_epoch_error)
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Época {epoch + 1}/{epochs}, Erro de Reconstrução: {avg_epoch_error:.6f}")
        
        return reconstruction_errors
    
    def transform(self, X):
        return self._sigmoid(np.dot(X, self.W) + self.c)