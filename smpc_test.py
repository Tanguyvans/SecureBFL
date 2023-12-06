import random
from flowerclient import FlowerClient
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def dataPreparation(filename, numberOfNodes=3): 
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
    df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
    df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    df = df.dropna()

    num_samples = len(df)

    split_size = num_samples // numberOfNodes

    multi_df = []
    for i in range(numberOfNodes):
        if i < numberOfNodes-1: 
            subset = df.iloc[i*split_size:(i+1)*split_size]
        else: 
            subset = df.iloc[i*split_size:]

        X_s = subset.drop(['satisfaction'],axis=1)
        y_s=subset[['satisfaction']]

        # X_s = df.drop(['satisfaction'],axis=1)
        # y_s=df[['satisfaction']]

        multi_df.append([X_s, y_s])

    return multi_df

def encrypt(x, n_shares=2):
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)

# Fonction pour appliquer le SMPC à une liste de listes
def apply_smpc(input_list, n_shares=2):
    encrypted_list = [[] for i in range(n_shares)]
    for inner_list in input_list:
        if isinstance(inner_list[0], np.ndarray):
            encrypted_inner_list = [[] for i in range(n_shares)]
            for inner_inner_list in inner_list: 
                encrypted_inner_inner_list = [[] for i in range(n_shares)]
                for x in inner_inner_list: 
                    crypted_tuple = encrypt(x, n_shares)
                    for i in range(n_shares):
                        encrypted_inner_inner_list[i].append(crypted_tuple[i])

                for i in range(n_shares):
                    encrypted_inner_list[i].append(encrypted_inner_inner_list[i])
        else: 
            encrypted_inner_list = [[] for i in range(n_shares)]
            for x in inner_list: 
                crypted_tuple = encrypt(x, n_shares)


                for i in range(n_shares):
                    encrypted_inner_list[i].append(crypted_tuple[i])
        
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_inner_list[i]))

    return encrypted_list
# Fonction pour décrypter une liste de listes chiffrées
def decrypt_list_of_lists(encrypted_list):
    decrypted_list = []
    n_shares = len(encrypted_list)

    for i in range(len(encrypted_list[0])): 
        sum_array = np.add(encrypted_list[0][i], encrypted_list[1][i])
        for j in range(2, n_shares): 
            sum_array = np.add(sum_array, encrypted_list[j][i])

        decrypted_list.append(sum_array)

    return decrypted_list

train_path = 'Airline Satisfaction/train.csv'
test_path = 'Airline Satisfaction/test.csv'
n_shares = 3


train_sets = dataPreparation(train_path, 1)
test_sets = dataPreparation(test_path, 1)
X_train, y_train = train_sets[0]
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
X_test, y_test = test_sets[0]
flower_client = FlowerClient(256, X_train, X_val ,X_test, y_train, y_val, y_test)

old_params = flower_client.get_parameters({})
res = old_params[:]
for i in range(1):
    res = flower_client.fit(res, {})[0]
    loss = flower_client.evaluate(res, {})[0]

encrypted_result = apply_smpc(res, n_shares)
decrypted_result = decrypt_list_of_lists(encrypted_result)

print("Liste de listes originale :")
print(res)

# print("\nListe de listes chiffrée :")
# print(encrypted_result[0])
# print(encrypted_result[1])
# print(encrypted_result[2])

print("\nListe de listes déchiffré :")
print(decrypted_result)