from ucimlrepo import fetch_ucirepo 
from sklearn.utils import shuffle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

iris = fetch_ucirepo(id=53) 

X = iris.data.features
y = iris.data.targets

df = pd.concat([X, y], axis=1)
df['species_category'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df = shuffle(df)

num_samples = len(df)
split_size = num_samples // 3

subset1 = df.iloc[:split_size]
subset2 = df.iloc[split_size:2*split_size]
subset3 = df.iloc[2*split_size:]

print(subset1.head())

subset1['species_category'] = subset1['species_category'].replace({1: 2, 0: 1, 2: 0})

print(subset1.head())
