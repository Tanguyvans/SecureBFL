from node import NodeClient, NodeServer
import pandas as pd
import random

import warnings
warnings.filterwarnings("ignore")

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

        multi_df.append([X_s, y_s])

    return multi_df

def nodePoisonning(poisonned_number, train_sets, test_sets): 
    for i in range(poisonned_number): 
        train_sets[i][1] = train_sets[i][1].replace({0: 1, 1: 0})
        test_sets[i][1] = test_sets[i][1].replace({0: 1, 1: 0})
        
    return train_sets[:], test_sets[:]
    
def nodeCreation(train_sets, test_sets, batch_size, coef_usefull = 1.02):
    nodes = []
    for i in range(len(train_sets)):
        nodeServer = NodeServer(f"111{i+1}", f"node{i+1}", batch_size, train_sets[i], test_sets[i], coef_usefull)
        nodeClient = NodeClient(nodeServer)
        nodes.append((nodeServer, nodeClient))

    return nodes

def startConnections(nodes): 
    nodeServers = [nodeServer for (nodeServer, nodeClient) in nodes]
    for i in range(len(nodes)):
        nodes[i][1].clientConnection(nodeServers)

def createAndBroadcastGlobalModel(node): 
    nodeServer = node[0]
    nodeClient = node[1]
    block = nodeServer.CreateGlobalModel()
    nodeClient.broadcast_block(block)
    nodeServer.AppendBlock(block)
    nodeClient.broadcast_decision()

def updateAndBroadcastModel(node):
    nodeServer = node[0]
    nodeClient = node[1]
    server_msg, block = nodeServer.UpdateModel()
    msg = nodeClient.broadcast_block(block)
    msg['tot']+=1
    if server_msg == True:
        msg['success']+=1
    else: 
        msg['failure']+=1
    print(msg)
    if msg['success']/msg['tot'] >= 2/3: 
        nodeServer.AppendBlock(block)
        nodeClient.broadcast_decision()

def saveNodesChain(nodes): 
    for node in nodes: 
        node[0].blockchain.save_chain_in_file(node[0].id)

if __name__ == "__main__":

    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'
    numberOfUpdatesRequired = 3
    numberOfNodes = 6
    poisonned_number = 2
    epochs = 10

    with open("output.txt", "w") as f: 
            f.write("")

    train_sets = dataPreparation(train_path, numberOfNodes)
    test_sets = dataPreparation(test_path, numberOfNodes)

    for i in range(poisonned_number): 
        train_sets[i][1] = train_sets[i][1].replace({0: 1, 1: 0})
        test_sets[i][1] = test_sets[i][1].replace({0: 1, 1: 0})

    nodes = nodeCreation(train_sets, test_sets, batch_size=256, coef_usefull=1.02)
    startConnections(nodes)

    createAndBroadcastGlobalModel(nodes[0])
    for i in range(epochs): 
        print(f"EPOCH: {i+1}")
        for node in nodes: 
            updateAndBroadcastModel(node)

        answer = nodes[0][0].aggregateParameters(numberOfUpdatesRequired)

        if answer["status"] == True:
            msg = nodes[0][1].broadcast_block(answer["message"]) 
            nodes[0][0].AppendBlock(answer["message"])
            nodes[0][1].broadcast_decision()
        else: 
            print(answer["message"]) 

    saveNodesChain(nodes)
