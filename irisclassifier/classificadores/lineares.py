# lineares.py
import numpy as np
import sympy as sp

def distancia_euclidiana(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def funcao_decisao(vetor_caracteristica, vetor_media_classe):
    return np.dot(np.transpose(vetor_media_classe), vetor_caracteristica) - np.dot(0.5 * np.transpose(vetor_media_classe), vetor_media_classe)

def distancia_minima_geral(sample, class_means):
    distances = {name: distancia_euclidiana(sample, mean) for name, mean in class_means.items()}
    return min(distances, key=distances.get)

def distancia_maxima_geral(sample, class_means):
    scores = {name: funcao_decisao(sample, mean) for name, mean in class_means.items()}
    return max(scores, key=scores.get)

def get_linear_decision_surface_equation(mean1, mean2, num_features):
    x_symbols = sp.symbols(f'x1:{num_features + 1}')
    x = sp.Matrix(x_symbols)
    
    # Converte médias para o formato do sympy
    m1 = sp.Matrix(mean1.flatten())
    m2 = sp.Matrix(mean2.flatten())
    
    # Função de decisão simbólica para a Classe 1: d1 = m1^T * x - 0.5 * m1^T * m1
    d1 = (m1.T * x)[0] - 0.5 * (m1.T * m1)[0]
    
    # Função de decisão simbólica para a Classe 2
    d2 = (m2.T * x)[0] - 0.5 * (m2.T * m2)[0]
    
    # A superfície de decisão é onde d1 - d2 = 0
    decision_surface = sp.simplify(d1 - d2)
    
    # Arredonda a expressão para melhor visualização
    def round_expr(expr, num_digits):
        return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})

    rounded_eq = round_expr(decision_surface, 3)
    
    return str(rounded_eq)
