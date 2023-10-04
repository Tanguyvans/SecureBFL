from ucimlrepo import fetch_ucirepo 
from sklearn.utils import shuffle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

iris = fetch_ucirepo(id=2) 

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

X = subset1.iloc[:, 0:4].values
y = subset1.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)