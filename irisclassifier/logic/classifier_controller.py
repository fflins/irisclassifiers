# logic/classifier_controller.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import combinations

# Importando seus módulos de classificadores
from classificadores import lineares, bayes, perceptron, mlp, rbm, kmeans
# Importando seu data_loader
import data_loader


def run_linear_classifier(classifier_type: str):
    """
    Executa os classificadores lineares e também calcula as superfícies de decisão.
    """
    # Usando o data_loader original para carregar os dados do Iris
    _, _, test_sample, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
    
    test_data = test_sample.drop(columns="Species").values
    true_labels = test_sample["Species"].values
    
    # Organiza as médias em um dicionário para facilitar o acesso
    class_means = {
        "setosa": setosasMean,
        "versicolor": versicolorMean,
        "virginica": virginicaMean
    }
    class_names = list(class_means.keys())
    num_features = test_data.shape[1]

    # Usa as funções genéricas para predição
    if classifier_type == "minimal":
        predictions = [lineares.distancia_minima_geral(sample, class_means) for sample in test_data]
    elif classifier_type == "maximal":
        predictions = [lineares.distancia_maxima_geral(sample, class_means) for sample in test_data]
    else:
        raise ValueError("Tipo de classificador linear desconhecido.")

    accuracy = np.mean(predictions == true_labels)

    # --- NOVA PARTE: Calcula as superfícies de decisão ---
    decision_surfaces = {}
    for c1_name, c2_name in combinations(class_names, 2):
        mean1 = class_means[c1_name]
        mean2 = class_means[c2_name]
        equation = lineares.get_linear_decision_surface_equation(mean1, mean2, num_features)
        decision_surfaces[f"{c1_name} vs {c2_name}"] = equation
    
    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "accuracy": accuracy,
        "class_names": class_names,
        "decision_surfaces": decision_surfaces 
    }

def run_bayes_classifier():
    """
    Executa o classificador Bayesiano (QDA).
    """
    _, _, test_sample, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
    test_data = test_sample.drop(columns="Species").values
    true_labels = test_sample["Species"].values

    cov_setosas = np.cov(setosasData, rowvar=False, bias=True)
    cov_versicolor = np.cov(versicolorData, rowvar=False, bias=True)
    cov_virginica = np.cov(virginicaData, rowvar=False, bias=True)

    predictions = [bayes.predict_bayes(s, cov_setosas, cov_versicolor, cov_virginica, setosasMean, versicolorMean, virginicaMean) for s in test_data]
    accuracy = np.mean(predictions == true_labels)

    # Gerar equações da superfície de decisão
    surfaces = {
        "Setosa vs Versicolor": bayes.print_decision_surface_equation(cov_setosas, cov_versicolor, setosasMean, versicolorMean),
        "Setosa vs Virginica": bayes.print_decision_surface_equation(cov_setosas, cov_virginica, setosasMean, virginicaMean),
        "Versicolor vs Virginica": bayes.print_decision_surface_equation(cov_versicolor, cov_virginica, versicolorMean, virginicaMean)
    }

    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "accuracy": accuracy,
        "decision_surfaces": surfaces
    }

# --- Funções para o Perceptron ---

def run_perceptron_training(class1: str, class2: str, rule_type: str, alpha: float, max_iter: int):
    X_train, y_train, X_test, y_test = data_loader.load_perceptron_data(class1, class2)
    
    if rule_type == 'Normal':
        weights, errors = perceptron.train_perceptron(X_train, y_train, alpha=alpha, max_iterations=max_iter)
    elif rule_type == 'Delta':
        weights, errors = perceptron.train_perceptron_delta_rule(X_train, y_train, alpha=alpha, max_iterations=max_iter)
    else:
        raise ValueError("Tipo de perceptron desconhecido.")

    # Realiza predições no conjunto de teste
    raw_predictions = np.dot(X_test, weights)
    predictions_numeric = np.where(raw_predictions >= 0, 1, -1)
    
    # Calcula acurácia
    accuracy = np.mean(predictions_numeric == y_test)
    
    # Converte rótulos numéricos de volta para strings para avaliação
    predictions_str = [class1 if p == 1 else class2 for p in predictions_numeric]
    expected_str = [class1 if t == 1 else class2 for t in y_test]

    return {
        "weights": weights,
        "errors": errors,
        "predictions": predictions_str,
        "expected": expected_str,  # <- nome corrigido aqui
        "accuracy": accuracy
    }



def to_one_hot(y_integers, num_classes):
    return np.eye(num_classes)[y_integers]

def run_mlp_training(hidden_size=10, epochs=10000, learning_rate=0.1):
    """
    Executa o ciclo completo de treinamento e teste para o MLP.
    AGORA RETORNA TAMBÉM O SCALER E OS NOMES DAS CLASSES.
    """
    iris = load_iris()
    X = iris.data
    # Usamos iris.target para obter os índices (0,1,2) e target_names para os nomes
    y_integers = iris.target
    y_names = iris.target_names
    
    # 1. Pré-processamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_one_hot = to_one_hot(y_integers, len(y_names))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=42)

    # 2. Treinamento
    model = mlp.MLP(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=len(y_names))
    loss_history = model.train(X_train, y_train, epochs, learning_rate)

    # 3. Avaliação
    y_pred_probs = model.forward(X_test)
    predictions = np.argmax(y_pred_probs, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)

    return {
        "model": model,
        "scaler": scaler,
        "class_names": y_names,
        "loss_history": loss_history,
        "predictions": predictions,
        "true_labels": true_labels,
        "accuracy": accuracy
    }

def run_rbm_training(num_hidden=20, epochs=300, learning_rate=0.001):
    """
    Executa o treinamento da RBM Gaussiana-Bernoulli.
    """
    iris = load_iris()
    X, y_true = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Instanciar e treinar o novo modelo
    model = rbm.RBM(num_visible=X_scaled.shape[1], num_hidden=num_hidden)
    reconstruction_errors = model.train(X_scaled, epochs=epochs, learning_rate=learning_rate, batch_size=16)

    # 3. Extrair características aprendidas
    learned_features = model.transform(X_scaled)

    return {
        "model": model,
        "reconstruction_errors": reconstruction_errors,
        "learned_features": learned_features,
        "original_labels": y_true, # Para a visualização
        "scaled_data": X_scaled # Para a visualização
    }

def run_kmeans_clustering(k=3, max_iters=100):
    """
    Executa o algoritmo K-Means no dataset Iris.
    """
    iris = load_iris()
    X = iris.data
    y_true = iris.target # Rótulos verdadeiros, usados APENAS para avaliação final

    # 1. Normalizar os dados é crucial para algoritmos baseados em distância
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Instanciar e treinar o modelo K-Means
    model = kmeans.KMeans(k=k, max_iters=max_iters)
    model.fit(X_scaled)


    return {
        "scaled_data": X_scaled,
        "true_labels": y_true,
        "cluster_labels": model.labels_,
        "centroids": model.centroids,
        "k": k
    }