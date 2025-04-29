import data_loader as data_loader
import numpy as np

def perceptron(classe1, classe2, alpha=0.01, max_iterations=100):
    values, classes = data_loader.join_classes(classe1, classe2)
    
    # separar os dados por classe
    classe1_indices = np.where(classes == 1)[0]
    classe2_indices = np.where(classes == -1)[0]
    
    # calcular o número de amostras para treinamento para cada classe (70%)
    train_size_classe1 = int(len(classe1_indices) * 0.7)
    train_size_classe2 = int(len(classe2_indices) * 0.7)
    
    # definir a semente para reprodutibilidade
    np.random.seed(50)
    
    # embaralhar os índices de cada classe
    np.random.shuffle(classe1_indices)
    np.random.shuffle(classe2_indices)
    
    # dividir os índices em treino e teste para cada classe
    classe1_train_idx = classe1_indices[:train_size_classe1]
    classe1_test_idx = classe1_indices[train_size_classe1:]
    
    classe2_train_idx = classe2_indices[:train_size_classe2]
    classe2_test_idx = classe2_indices[train_size_classe2:]
    
    # combinar os índices de treino e teste
    train_idx = np.concatenate([classe1_train_idx, classe2_train_idx])
    test_idx = np.concatenate([classe1_test_idx, classe2_test_idx])
    
    # embaralhar os índices de treino para não ter todas as amostras de uma classe seguidas
    np.random.shuffle(train_idx)
    
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

        # média dos erros da época
        avg_error = np.mean(epoch_errors)
        errors.append(avg_error)


    return weights, errors, values_test, classes_test