# bayes.py
import numpy as np
import pandas as pd

def calculate_covariance_matrix(class_data, mean_vector):
    n = class_data.shape[0]
    cov_matrix = np.zeros((class_data.shape[1], class_data.shape[1]))
    
    mean_vector = mean_vector.reshape(-1, 1)
    
    for i in range(n):
        x = class_data.iloc[i, :].values.reshape(-1, 1)
        diff = x - mean_vector
        cov_matrix += np.dot(diff, diff.T)
    
    cov_matrix = cov_matrix / n
    
    return cov_matrix

def inverse_cov_matrix(cov_matrix):
    try:
        inverse = np.linalg.inv(cov_matrix)
        return inverse
    except np.linalg.LinAlgError:
        print("A matriz de covariância não é inversível (determinante é zero).")
        return None
    

def calculate_decision_function(sample, mean_vector, cov_matrix): 
    
    mean_vector = mean_vector.reshape(-1, 1)
    sample = sample.reshape(-1, 1)
    
    # Calculate the inverse of the covariance matrix
    inv_cov = inverse_cov_matrix(cov_matrix)
    
    # Calculate the determinant of the covariance matrix
    det_cov = np.linalg.det(cov_matrix)
    
    # Calculate the difference between the sample and the mean vector
    diff = sample - mean_vector
    
    # Calculate the decision function using the formula provided
    decision = np.log(1/3) - 0.5 * np.log(det_cov) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff))

    return decision[0, 0]


def predict_bayes(sample, setosa_matrix, versicolor_matrix, virginica_matrix, mean_setosas, mean_versicolor, mean_virginica):
    decision_setosas = calculate_decision_function(sample, mean_setosas, setosa_matrix)
    decision_versicolor = calculate_decision_function(sample, mean_versicolor, versicolor_matrix)
    decision_virginica = calculate_decision_function(sample, mean_virginica, virginica_matrix)
    
    if decision_setosas > decision_versicolor and decision_setosas > decision_virginica:
        return "setosa"
    elif decision_versicolor > decision_setosas and decision_versicolor > decision_virginica:
        return "versicolor"
    else:
        return "virginica"


