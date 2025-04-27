import numpy as np
from sklearn.metrics import confusion_matrix

# calcula o coeficiente kappa
def kappa(expected, predictions):
    # cria a matriz de confusão
    cm = confusion_matrix(expected, predictions)
    # total de amostras
    total_samples = len(expected)
    # acurácia geral
    accuracy = np.trace(cm) / total_samples
    # acurácia aleatória
    accuracy_random = random_accuracy(total_samples, cm)


    kappa = (accuracy - accuracy_random) / (1 - accuracy_random)
    return kappa

# calcula a variância do coeficiente kappa
def kappa_variance(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    # se a acurácia for 1, a variância é 0, então retorna logo pra não dividir por 0
    if accuracy == 1: return 0
    accuracy_random = random_accuracy(total_samples, cm)


    # calcula o número de classes únicas
    num_classes = len(np.unique(expected))   

    # calcula o terceiro termo da variância
    third_term=0
    for i in range(num_classes):
        #pra cada elemento da diagonal, adiciona a soma da linha e da coluna multiplicada pelo elemento da diagonal (aii * (a+i + ai+))
        third_term += (np.sum(cm[i, :]) + np.sum(cm[:, i])) * cm[i, i]
    third_term = third_term / (total_samples ** 2)

    # calcula o quarto termo da variância
    fourth_term=0
    for i in range(num_classes):
        for j in range(num_classes):
            #pra cada elemento da matriz, adiciona a soma da linha e da coluna multiplicado pelo elemento atual 
            fourth_term += ((np.sum(cm[i, :]) + np.sum(cm[:, j])) ** 2) * cm[i, i] 
    fourth_term = fourth_term / (total_samples** 3)


    # divide a formula em 3 partes pra facilitar
    kappa_variance_1 = (accuracy *(1 - accuracy))/(1-accuracy)**2
    kappa_variance_2 = ((2 *(1-accuracy)) * (2*accuracy*accuracy_random - third_term))/(1-accuracy_random)**3
    kappa_variance_3 = (1-accuracy)**2 * (fourth_term - 4 * accuracy_random)**2 / 1-accuracy_random**4

    # calcula a variância
    kappa_variance = kappa_variance_1 + kappa_variance_2 + kappa_variance_3
    return kappa_variance/total_samples

# calcula o coeficiente tau
def tau(expected, predictions):
    cm = confusion_matrix(expected,predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    tau = accuracy - 1/len(np.unique(expected))
    tau = tau / (1 - 1/len(np.unique(expected)))
    return tau

# calcula a variância do coeficiente tau
def tau_variance(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    tau_variance = accuracy * (1 - accuracy) / (1 - 1/len(np.unique(expected)))**2
    tau_variance = tau_variance / (total_samples)
    return tau_variance

# calcula a precisão para cada classe
def precision(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    true_positives = cm[0, 0]  
    false_positives = cm[0, 1] 
    precision = true_positives / (true_positives + false_positives)
    return precision

# calcula a revocaçao para cada classe
def recall(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    true_positives = cm[0, 0]
    false_negatives = cm[1, 0]
    recall = true_positives / (true_positives + false_negatives)
    return recall

# calcula o f1-score para cada classe
def f1_score(expected, predictions):
    # calcula a precisão e a revocação
    precision_ = precision(expected, predictions)
    recall_ = recall(expected, predictions)
 
        # calcula o f1-score
    f1_score = 2 * ((precision_ * recall_) / (precision_ + recall_))
    return f1_score


# calcula a acurácia aleatória
def random_accuracy(total_samples, confusion_matrix):
    cm = confusion_matrix
    aux = 0
    # itera sobre as classes
    for i in range(len(cm)):
        # calcula a soma das linhas e colunas
        row_sum = np.sum(cm[i, :])
        col_sum = np.sum(cm[:, i])
        # acumula o produto das somas
        aux += row_sum * col_sum
    # calcula a acurácia aleatória
    return aux / (total_samples ** 2)