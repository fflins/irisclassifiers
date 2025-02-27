import pandas as pd
import numpy as np

data = pd.read_csv("data.csv", decimal=",")

data = data[data["Species"].isin(["setosa", "versicolor"])]

training_sample = data.sample(frac=0.7, random_state=12)
test_sample = data.drop(training_sample.index)

setosas = training_sample[training_sample["Species"] == "setosa"]
versicolor = training_sample[training_sample["Species"] == "versicolor"]

setosasData = setosas.drop(columns="Species")
versicolorData = versicolor.drop(columns="Species")

setosasMean = setosasData.mean().values
versicolorMean = versicolorData.mean().values

print("Setosa:", setosasMean)
print("Versicolor:", versicolorMean)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def min_distance_classifier(sample):
    dist_setosa = euclidean_distance(sample, setosasMean)
    dist_versicolor = euclidean_distance(sample, versicolorMean)
        
    if (dist_setosa< dist_versicolor): return "setosa"
    else: return "versicolor"
    
example = test_sample.iloc[20]
example_data = example.drop('Species').values
true_class = example['Species']
    
print(true_class)
print(min_distance_classifier(example_data))
    