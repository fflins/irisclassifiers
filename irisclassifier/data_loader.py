# data_loader.py
import pandas as pd
import numpy as np

def load_data():
    """Carrega e prepara os dados para os classificadores de mínima distância."""
    data = pd.read_csv("./data.csv", decimal=",")
    
    # separando os dados por classe
    setosa = data[data["Species"] == "setosa"]
    versicolor = data[data["Species"] == "versicolor"]
    virginica = data[data["Species"] == "virginica"]
    
    train_size = int(len(setosa) * 0.7)
    
    # criando amostras de treinamento
    setosa_train = setosa.sample(n=train_size, random_state=12)
    versicolor_train = versicolor.sample(n=train_size, random_state=12)
    virginica_train = virginica.sample(n=train_size, random_state=12)
    
    # criando amostras de teste 
    setosa_test = setosa.drop(setosa_train.index)
    versicolor_test = versicolor.drop(versicolor_train.index)
    virginica_test = virginica.drop(virginica_train.index)
    
    training_sample = pd.concat([setosa_train, versicolor_train, virginica_train])
    test_sample = pd.concat([setosa_test, versicolor_test, virginica_test])
    
    setosasData = setosa_train.drop(columns="Species")
    versicolorData = versicolor_train.drop(columns="Species")
    virginicaData = virginica_train.drop(columns="Species")
    
    setosasMean = setosasData.mean().values
    versicolorMean = versicolorData.mean().values
    virginicaMean = virginicaData.mean().values

    return data, training_sample, test_sample, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean

def load_perceptron_data(class1, class2, train_split=0.7, random_state=50):

    data = pd.read_csv("./data.csv", decimal=",")
    data_filtered = data[data["Species"].isin([class1, class2])]
    
    values = data_filtered.drop(columns=["Species"]).values
    classes = np.where(data_filtered["Species"] == class1, 1, -1)
    
    values = np.hstack([values, np.ones((values.shape[0], 1))])
    
    class1_indices = np.where(classes == 1)[0]
    class2_indices = np.where(classes == -1)[0]

    train_size_class1 = int(len(class1_indices) * train_split)
    train_size_class2 = int(len(class2_indices) * train_split)

    np.random.seed(random_state)
    np.random.shuffle(class1_indices)
    np.random.shuffle(class2_indices)

    class1_train_idx = class1_indices[:train_size_class1]
    class1_test_idx = class1_indices[train_size_class1:]
    
    class2_train_idx = class2_indices[:train_size_class2]
    class2_test_idx = class2_indices[train_size_class2:]
    
    train_idx = np.concatenate([class1_train_idx, class2_train_idx])
    test_idx = np.concatenate([class1_test_idx, class2_test_idx])
    
    np.random.shuffle(train_idx) 

    X_train = values[train_idx]
    y_train = classes[train_idx]
    X_test = values[test_idx]
    y_test = classes[test_idx]
    
    return X_train, y_train, X_test, y_test
