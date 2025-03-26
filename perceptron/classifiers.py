# classifiers.py
import numpy as np
import sympy as sp

def distancia_euclidiana(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def distancia_minima(sample, setosasMean, versicolorMean, virginicaMean):

    dist_setosa = distancia_euclidiana(sample, setosasMean)
    dist_versicolor = distancia_euclidiana(sample, versicolorMean)
    dist_virginica = distancia_euclidiana(sample, virginicaMean)
    
    if dist_setosa < dist_versicolor and dist_setosa < dist_virginica:
        return "setosa"
    elif dist_virginica < dist_versicolor and dist_virginica < dist_setosa:
        return "virginica"
    else:
        return "versicolor"

def funcao_decisao(vetor_caracteristica, vetor_media_classe):
    return np.dot(np.transpose(vetor_media_classe), vetor_caracteristica) - \
           np.dot(0.5 * np.transpose(vetor_media_classe), vetor_media_classe)

def distancia_maxima(sample, setosasMean, versicolorMean, virginicaMean):
    dist_setosa = funcao_decisao(sample, setosasMean)
    dist_versicolor = funcao_decisao(sample, versicolorMean)
    dist_virginica = funcao_decisao(sample, virginicaMean)
    
    if dist_setosa > dist_versicolor and dist_setosa > dist_virginica:
        return "setosa"
    elif dist_virginica > dist_versicolor and dist_virginica > dist_setosa:
        return "virginica"
    else:
        return "versicolor"

    
def superficie_decisao(media1, media2):
    a = 2 * (media2 - media1)
    b = np.dot(media1, media1) - np.dot(media2, media2)
    return a, b

