import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def load_data():
    """
    Carrega os dados e faz a divisão estratificada garantindo o mesmo número de amostras
    de cada classe nos conjuntos de treino e teste.
    """
    data = pd.read_csv("../data.csv", decimal=",")
    
    # separando os dados por classe
    setosa = data[data["Species"] == "setosa"]
    versicolor = data[data["Species"] == "versicolor"]
    virginica = data[data["Species"] == "virginica"]
    
    train_size = int(len(setosa) * 0.7)
    
    # criando amostras de treinamento
    setosa_train = setosa.sample(n=train_size, random_state=5)
    versicolor_train = versicolor.sample(n=train_size, random_state=5)
    virginica_train = virginica.sample(n=train_size, random_state=5)
    
    # criando amostras de teste 
    setosa_test = setosa.drop(setosa_train.index)
    versicolor_test = versicolor.drop(versicolor_train.index)
    virginica_test = virginica.drop(virginica_train.index)
    
    # combinando amostras de treinamento e teste
    training_sample = pd.concat([setosa_train, versicolor_train, virginica_train])
    test_sample = pd.concat([setosa_test, versicolor_test, virginica_test])
    
    # separando dados por classe para cálculos de média
    setosasData = setosa_train.drop(columns="Species")
    versicolorData = versicolor_train.drop(columns="Species")
    virginicaData = virginica_train.drop(columns="Species")
    
    # Calculando médias
    setosasMean = setosasData.mean().values
    versicolorMean = versicolorData.mean().values
    virginicaMean = virginicaData.mean().values

    return data, training_sample, test_sample, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean

def join_classes(classe1, classe2):
    data = pd.read_csv("../data.csv", decimal=",")
    data = data[data["Species"].isin([classe1, classe2])]
    values = data.drop(columns=["Species"]).values  
    classes = np.array([1 if label == classe1 else -1 for label in data["Species"]]) 
    values = np.hstack([values, np.ones((values.shape[0], 1))])
    
    return values, classes  

load_data()