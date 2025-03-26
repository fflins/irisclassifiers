import pandas as pd
import numpy as np

def load_data():

    data = pd.read_csv("../data.csv", decimal=",")  
      
    training_sample = data.sample(frac=0.7, random_state=12)
    test_sample = data.drop(training_sample.index)
    
    training_size = len(training_sample)
    test_size = len(test_sample)
    print(f"Training set size: {training_size}")
    print(f"Test set size: {test_size}")
    
    setosas = training_sample[training_sample["Species"] == "setosa"]
    versicolor = training_sample[training_sample["Species"] == "versicolor"]
    virginica = training_sample[training_sample["Species"] == "virginica"]
    
    setosasData = setosas.drop(columns="Species")
    versicolorData = versicolor.drop(columns="Species")
    virginicaData = virginica.drop(columns="Species")
    
    setosasMean = setosasData.mean().values
    versicolorMean = versicolorData.mean().values
    virginicaMean = virginicaData.mean().values
    
    return data, training_sample, test_sample, setosasMean, versicolorMean, virginicaMean

def join_classes(classe1, classe2):
    data = pd.read_csv("../data.csv", decimal=",")
    data = data[data["Species"].isin([classe1, classe2])]
    values = data.drop(columns=["Species"]).values  
    classes = np.array([1 if label == classe1 else -1 for label in data["Species"]]) 
    values = np.hstack([values, np.ones((values.shape[0], 1))])
    
    return values, classes  
