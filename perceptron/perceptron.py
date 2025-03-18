import data_loader
import numpy as np

def perceptron(classe1, classe2, alpha=0.01, max_iterations=100):
    values, classes = data_loader.join_classes(classe1, classe2)

    #separar treino e teste
    np.random.seed(50)
    indices = np.random.permutation(len(values))
    split = int(0.7 * len(values))
    train_idx, test_idx = indices[:split], indices[split:]
    
    values_train = values[train_idx]
    classes_train = classes[train_idx]
    values_test = values[test_idx]
    classes_test = classes[test_idx]

    #vetor peso
    weights = np.random.rand(values.shape[1])  

    errors = []
    
    #iterar epocas
    for epoch in range(max_iterations):
        epoch_errors = []
        
        #iterar amostras
        for i in range(len(values_train)):
            sample = values_train[i]
            label = classes_train[i]

            output = np.dot(weights, sample)
            #funçao ativaçao
            predicted_class = 1 if output >= 0 else -1
            
            #erro E(w) = (1/2)*(r-w^Tx)²
            error = 0.5 * (label - output)**2
            epoch_errors.append(error)
            
            if predicted_class != label:
                #regra delta alpha(r - wt * x) * x
                weights += alpha * (label - predicted_class) * sample

        # Média dos erros da época
        avg_error = np.mean(epoch_errors)
        errors.append(avg_error)


    return weights, errors, values_test, classes_test