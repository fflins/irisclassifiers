import data_loader
import numpy as np

def perceptron(classe1, classe2,alpha=0.01, max_iterations=100):
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


    accuracies = []  
    
    #iterar epocas
    for epoch in range(max_iterations):
        #iterar amostras
        for i in range(len(values_train)):
            sample = values_train[i]
            label = classes_train[i]

            output = np.dot(weights, sample)
            #funçao ativaçao
            predicted_class = 1 if output >= 0 else -1
            
            if predicted_class != label:
                #regra delta alpha(r - wt * x) * x
                weights += alpha * (label - predicted_class) * sample


        correct = 0
        for i in range(len(values_test)):
            output = np.dot(weights, values_test[i])
            prediction = 1 if output >= 0 else -1
            if prediction == classes_test[i]:
                correct += 1

        accuracy = correct / len(values_test)
        accuracies.append(accuracy)  

    return weights, accuracies, values_test, classes_test