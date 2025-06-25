# perceptron.py
import numpy as np

def train_perceptron(X_train, y_train, alpha=0.01, max_iterations=100):

    weights = np.random.rand(X_train.shape[1])
    print(weights)
    mse_history = [] 

    for _ in range(max_iterations):
        epoch_squared_errors = []
        for sample, label in zip(X_train, y_train):
            output = np.dot(weights, sample)
            
            squared_error = 0.5 * (label - output)**2
            epoch_squared_errors.append(squared_error)

            predicted_class = 1 if output >= 0 else -1
            
            if predicted_class<label:
                weights += alpha  * sample
            elif predicted_class > label:
                weights -= alpha * sample
            print(weights)
            
        
        mse_history.append(np.mean(epoch_squared_errors))

    return weights, mse_history

def train_perceptron_delta_rule(X_train, y_train, alpha=0.01, max_iterations=100):

    weights = np.random.rand(X_train.shape[1])
    print(weights)

    mse_history = [] 

    for _ in range(max_iterations):
        epoch_squared_errors = []
        for sample, label in zip(X_train, y_train):
            output = np.dot(weights, sample)
            squared_error = 0.5 * (label - output)**2
            epoch_squared_errors.append(squared_error)
            
        
            update = alpha * (label - output) * sample
            weights += update
            print(weights)

            
        mse_history.append(np.mean(epoch_squared_errors))

    return weights, mse_history