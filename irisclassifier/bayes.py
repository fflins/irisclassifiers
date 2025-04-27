# bayes.py
import numpy as np
import sympy as sp

def calculate_covariance_matrix(class_data, mean_vector):
    n = class_data.shape[0]
    cov_matrix = np.zeros((class_data.shape[1], class_data.shape[1]))

    mean_vector = mean_vector.reshape(-1, 1)
    mean_vector_transposed = np.transpose(mean_vector)
    
    for i in range(n):
        x = class_data.iloc[i, :].values.reshape(-1, 1)
        x_transposed = np.transpose(x)
        cov_matrix += np.dot(x,x_transposed) - np.dot(mean_vector, mean_vector_transposed)
    
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
    
    inv_cov = inverse_cov_matrix(cov_matrix)

    det_cov = np.linalg.det(cov_matrix)

    #(x - mj)
    diff = sample - mean_vector

    decision = - 0.5 * np.log(det_cov) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) + np.log(1/3) 


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

def print_decision_surface_equation(class1_cov, class2_cov, class1_mean, class2_mean):
    # criar símbolos para as 4 caracteristicas
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
    x = sp.Matrix([x1, x2, x3, x4])
    
    # Converter médias para sympy
    mean1 = sp.Matrix(class1_mean.flatten())
    mean2 = sp.Matrix(class2_mean.flatten())

    print("mean1", mean1)
    print("mean2", mean2)
    
    # Calcular determinantes e inversas
    inv_cov1 = sp.Matrix(inverse_cov_matrix(class1_cov))
    inv_cov2 = sp.Matrix(inverse_cov_matrix(class2_cov))
    det_cov1 = np.linalg.det(class1_cov)
    det_cov2 = np.linalg.det(class2_cov)
    
    # Calcular diferenças (x - μ)
    diff1 = x - mean1
    print("diff1", diff1)
    diff2 = x - mean2
    print("diff2", diff2)
    
    # Calcular as funções de decisão para cada classe
    # d(x) = -0.5 * ln(det(Σ)) - 0.5 * (x - μ)^T * Σ^(-1) * (x - μ) + ln(P(ω))
    term1_class1 = -0.5 * sp.log(det_cov1)
    print("term1_class1", term1_class1)
    term2_class1 = -0.5 * (diff1.T * inv_cov1 * diff1)[0]
    print("term2_class1", term2_class1)
    term3_class1 = sp.log(1/3) 
    print("term3_class1", term3_class1)
    d_class1 = term1_class1 + term2_class1 + term3_class1
    
    term1_class2 = -0.5 * sp.log(det_cov2)
    term2_class2 = -0.5 * (diff2.T * inv_cov2 * diff2)[0]
    term3_class2 = sp.log(1/3)  
    d_class2 = term1_class2 + term2_class2 + term3_class2
    
    # superfície de decisão é dada por d(classe1) - d(classe2) = 0
    decision_surface = d_class1 - d_class2
    print("decision_surface", decision_surface)
    
    # Simplificar a equação resultante
    simplified_eq = sp.simplify(decision_surface)
    print("simplified_eq", simplified_eq)
    
    # arredondar coeficientes para 2 casas decimais
    # primeiro, transformar para string para manipulação
    eq_str = str(simplified_eq)
    # função para substituir números na string da equação
    def round_numbers_in_eq(match):
        num = float(match.group(0))
        rounded = round(num, 2)
        # Evitar .0 no final dos números inteiros
        return str(rounded) if rounded != int(rounded) else str(int(rounded))
    
    import re
    # padrão para encontrar números na equação (inclusive decimais e negativos)
    pattern = r'-?\d+\.\d+'
    rounded_eq_str = re.sub(pattern, round_numbers_in_eq, eq_str)
    
    
    
    return rounded_eq_str
