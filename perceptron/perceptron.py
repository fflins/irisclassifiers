import data_loader
import numpy as np

def perceptron(classe1, classe2, features_indices):

    c = 0.1  
    max_iterations = 150  
    
    values, classes = data_loader.join_classes(classe1, classe2)
    #print("Valores antes da seleção:", values.shape)
    #print("Índices das features usados:", features_indices)
    values = values[:, features_indices]  
    #print("Valores depois da seleção:", values.shape)
    

    #adicionar 1 no final 
    values = np.hstack([values, np.ones((values.shape[0], 1))])
    #print(values)
    
    indices = list(range(len(values)))
    #print(indices)

    np.random.seed(12)
    np.random.shuffle(indices)


    #dividir 70% treino 30% teste
    split_idx = int(0.7 * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]


    values_train = values[train_indices]
    classes_train = classes[train_indices]
    values_test = values[test_indices]
    classes_test = classes[test_indices]
    
    # vetor pesos
    weights = np.zeros(values_train.shape[1])
    accuracy_history = []
    
    #iteraçoes totais
    for i in range(max_iterations):
        changes = 0
        #iteraçoes amostras
        for j in range(len(values_train)):
            output = np.dot(weights, values_train[j])
            predicted_class = 1 if output >= 0 else 0
            #print(output)
            #print(classes_train[j])
            if predicted_class != classes_train[j]:
                error = classes_train[j] - predicted_class
                weights += c * error * values_train[j]
                changes += 1

        train_predictions = np.array([1 if np.dot(weights, x) >= 0 else 0 for x in values_train])
        train_accuracy = calculate_accuracy(classes_train, train_predictions)
        accuracy_history.append(train_accuracy)

        if changes == 0:
            print(f"Convergiu em {i+1} iterações!")
            break
    
    # Evaluate on test set
    test_predictions = np.array([1 if np.dot(weights, x) >= 0 else 0 for x in values_test])
    test_accuracy = calculate_accuracy(classes_test, test_predictions)
    
    # Print results
    print(f"\nResultados para {classe1} vs {classe2}:")
    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    return weights, test_accuracy, accuracy_history

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy without using sklearn"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)