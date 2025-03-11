# classifiers.py
import numpy as np
import sympy as sp

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def minimal_distance_classifier(sample, setosasMean, versicolorMean, virginicaMean):

    dist_setosa = euclidean_distance(sample, setosasMean)
    dist_versicolor = euclidean_distance(sample, versicolorMean)
    dist_virginica = euclidean_distance(sample, virginicaMean)
    
    if dist_setosa < dist_versicolor and dist_setosa < dist_virginica:
        return "setosa"
    elif dist_virginica < dist_versicolor and dist_virginica < dist_setosa:
        return "virginica"
    else:
        return "versicolor"

def funcao_decisao(vetor_caracteristica, vetor_media_classe):
    return np.dot(np.transpose(vetor_media_classe), vetor_caracteristica) - \
           np.dot(0.5 * np.transpose(vetor_media_classe), vetor_media_classe)

def max_distance_classifier(sample, setosasMean, versicolorMean, virginicaMean):
    dist_setosa = funcao_decisao(sample, setosasMean)
    dist_versicolor = funcao_decisao(sample, versicolorMean)
    dist_virginica = funcao_decisao(sample, virginicaMean)
    
    if dist_setosa > dist_versicolor and dist_setosa > dist_virginica:
        return "setosa"
    elif dist_virginica > dist_versicolor and dist_virginica > dist_setosa:
        return "virginica"
    else:
        return "versicolor"

  
def evaluate_classifier(test_sample, classifier_func, setosasMean, versicolorMean, virginicaMean):
    hits = 0
    results = []
    
    for index, row in test_sample.iterrows():
        sample = row
        sample_data = sample.drop('Species').values
        true_class_sample = sample['Species']
        
        predicted_class = classifier_func(sample_data, setosasMean, versicolorMean, virginicaMean)
        
        if predicted_class == true_class_sample:
            hits += 1
            
        results.append({
            'true': true_class_sample,
            'predicted': predicted_class,
            'sample': sample_data
        })
    
    accuracy = (hits / len(test_sample)) * 100
    return accuracy, results

    
def get_decision_boundary_equation(media1, media2):
    a = 2 * (media2 - media1)
    b = np.dot(media1, media1) - np.dot(media2, media2)
    return a, b

