import numpy as np
from sklearn.metrics import confusion_matrix

def kappa(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    accuracy_random = random_accuracy(total_samples, cm)
    kappa = (accuracy - accuracy_random) / (1 - accuracy_random)
    return kappa



def kappa_variance(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    if accuracy == 1: return 0
    print("accuracy random kappa variance")
    accuracy_random = random_accuracy(total_samples, cm)

    num_classes = len(np.unique(expected))   

    third_term=0
    for i in range(num_classes):
        third_term += (np.sum(cm[i, :]) + np.sum(cm[:, i])) * cm[i, i]
    third_term = third_term / (total_samples ** 2)
    print(third_term)
    fourth_term=0
    for i in range(num_classes):
        for j in range(num_classes):
            
            fourth_term += ((np.sum(cm[i, :]) + np.sum(cm[:, j])) ** 2) * cm[i, i] 
    fourth_term = fourth_term / (total_samples** 3)
    print(fourth_term)
    kappa_variance_1 = (accuracy *(1 - accuracy))/(1-accuracy)**2
    kappa_variance_2 = ((2 *(1-accuracy)) * (2*accuracy*accuracy_random - third_term))/(1-accuracy_random)**3
    kappa_variance_3 = (1-accuracy)**2 * (fourth_term - 4 * accuracy_random)**2 / 1-accuracy_random**4

    kappa_variance = kappa_variance_1 + kappa_variance_2 + kappa_variance_3
    return kappa_variance/total_samples


def tau(expected, predictions):
    cm = confusion_matrix(expected,predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples
    print("unicos", len(np.unique(expected)))
    tau = accuracy - 1/len(np.unique(expected))
    tau = tau / (1 - 1/len(np.unique(expected)))
    return tau


def tau_variance(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    total_samples = len(expected)
    accuracy = np.trace(cm) / total_samples

    tau_variance = accuracy * (1 - accuracy) / (1 - 1/len(np.unique(expected)))**2
    tau_variance = tau_variance / (total_samples)
    return tau_variance


def significance(coef1, coef2, variance1, variance2):
    return coef1 - coef2 / np.sqrt(variance1 + variance2)


def precision(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    precision_values = []
    for i in range(len(cm)):
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - true_positives
        if true_positives + false_positives == 0:
            precision_values.append(0)  
        else:
            precision_values.append(true_positives / (true_positives + false_positives))
    return precision_values

def recall(expected, predictions):
    cm = confusion_matrix(expected, predictions)
    recall_values = []
    for i in range(len(cm)):
        true_positives = cm[i, i]
        false_negatives = np.sum(cm[i, :]) - true_positives
        if true_positives + false_negatives == 0:
            recall_values.append(0)  
        else:
            recall_values.append(true_positives / (true_positives + false_negatives))
    return recall_values







#funçao auxiliar
def random_accuracy(total_samples, confusion_matrix):
    cm = confusion_matrix
    aux = 0
    for i in range(len(cm)):
        row_sum = np.sum(cm[i, :])
        col_sum = np.sum(cm[:, i])
        aux += row_sum * col_sum

    return aux / (total_samples ** 2)

