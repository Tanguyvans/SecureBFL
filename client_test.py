import pandas as pd
from client import ClientClient, ClientServer
from node import NodeServer, NodeClient
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

        X_s = subset.drop(['satisfaction'], axis=1)
        y_s = subset[['satisfaction']]

        # X_s = df.drop(['satisfaction'],axis=1)
        # y_s=df[['satisfaction']]

        multi_df.append([X_s, y_s])

    return multi_df


def clientCreation(train_sets, test_sets, batch_size):
    clients = []
    for i in range(len(train_sets)):
        clientServer = ClientServer(f"111{i+1}", f"node{i+1}", batch_size, train_sets[i], test_sets[i])
        clientClient = ClientClient(clientServer)
        clients.append((clientServer, clientClient))

    return clients


def nodeCreation(nb, train_sets, test_sets, batch_size, coef_usefull = 1.02):
    nodes = []
    for i in range(nb):
        nodeServer = NodeServer(f"211{i+1}", f"node{i+1}", batch_size, train_sets[i], test_sets[i], coef_usefull)
        nodeClient = NodeClient(nodeServer)
        nodes.append((nodeServer, nodeClient))

    return nodes


def startClientNodeConnections(node, clients):
    nodeServer = node[0]
    nodeClient = node[1]
    clientServers = [clientServer for (clientServer, clientClient) in clients]
    for i in range(len(clients)):
        # start client to client connections
        clients[i][1].clientConnection(clientServers)
        clients[i][0].cluster = len(clientServers)
        # start client to node connections
        clients[i][1].clientNodeConnection(nodeServer)
        # start node to client connections
        nodeClient.clientConnection(clientServers)


def startNodeNodeConnections(nodes):
    nodeServers = [nodeServer for (nodeServer, nodeClient) in nodes]
    for i in range(len(nodes)):
        nodes[i][1].nodeConnection(nodeServers)

    return nodes


def createAndBroadcastGlobalModelToNodes(node):
    nodeServer = node[0]
    nodeClient = node[1]
    block = nodeServer.CreateGlobalModel()
    nodeClient.broadcast_block(block)
    nodeServer.AppendBlock(block)
    nodeClient.broadcast_decision()


def broadcastModelToClients(node):
    nodeServer = node[0]
    nodeClient = node[1]
    nodeClient.broadcast_global_model_to_clients(nodeServer.global_params_directory)


def updateAndBroadcastModel(node, weights):
    nodeServer = node[0]
    nodeClient = node[1]
    server_msg, block = nodeServer.UpdateModel(weights)
    msg = nodeClient.broadcast_block(block)
    msg['tot'] += 1
    if server_msg == True:
        msg['success'] += 1
    else:
        msg['failure'] += 1
    print(msg)
    if msg['success']/msg['tot'] >= 2/3:
        nodeServer.AppendBlock(block)
        nodeClient.broadcast_decision()


def saveNodesChain(nodes):
    for node in nodes:
        node[0].blockchain.save_chain_in_file(node[0].id)


if __name__ == "__main__":
    train_path = 'Data/Airline Satisfaction/train.csv'
    test_path = 'Data/Airline Satisfaction/test.csv'

    numberOfClients = 6
    poisonned_number = 0
    epochs = 25

    with open("output.txt", "w") as f:
        f.write("")

    train_sets = dataPreparation(train_path, numberOfClients)
    test_sets = dataPreparation(test_path, numberOfClients)

    clients = clientCreation(train_sets, test_sets, batch_size=256)
    nodes = nodeCreation(3, train_sets, test_sets, batch_size=256)

    startClientNodeConnections(nodes[0], [clients[0], clients[1]])
    nodes[0][0].createCluster([clients[0][0].id, clients[1][0].id])

    startClientNodeConnections(nodes[1], [clients[2], clients[3]])
    nodes[1][0].createCluster([clients[2][0].id, clients[3][0].id])

    startClientNodeConnections(nodes[2], [clients[4], clients[5]])
    nodes[2][0].createCluster([clients[4][0].id, clients[5][0].id])

    startNodeNodeConnections(nodes)
    createAndBroadcastGlobalModelToNodes(nodes[0])
    createAndBroadcastGlobalModelToNodes(nodes[1])
    createAndBroadcastGlobalModelToNodes(nodes[2])

    w0d = np.load(nodes[0][0].global_params_directory)
    w0 = [w0d[f'param_{i}'] for i in range(len(w0d)-1)]

    broadcastModelToClients(nodes[0])
    broadcastModelToClients(nodes[1])
    broadcastModelToClients(nodes[2])

    for i in range(poisonned_number):
        train_sets[i][1] = train_sets[i][1].replace({0: 1, 1: 0})
        test_sets[i][1] = test_sets[i][1].replace({0: 1, 1: 0})

    for i in range(epochs):
        print(f"EPOCH: {i+1}")
        # TRAINING

        clients[0][0].frag_weights = []
        clients[1][0].frag_weights = []
        clients[2][0].frag_weights = []
        clients[3][0].frag_weights = []
        clients[4][0].frag_weights = []
        clients[5][0].frag_weights = []

        frag_weights = clients[0][0].train()
        clients[0][1].sendFragmentedWeightsToClients(frag_weights)
        frag_weights = clients[1][0].train()
        clients[1][1].sendFragmentedWeightsToClients(frag_weights)

        nodes[0][0].cluster_weights = []
        for k, v in nodes[0][0].cluster.items():
            nodes[0][0].cluster[k] = 0

        clients[0][1].sendWeightsToNode(clients[0][0].sum_weights, 10)
        clients[1][1].sendWeightsToNode(clients[1][0].sum_weights, 10)

        weights = nodes[0][0].clusterAggregation()

        updateAndBroadcastModel(nodes[0], weights)

        print("done node 1")

        frag_weights = clients[2][0].train()
        clients[2][1].sendFragmentedWeightsToClients(frag_weights)
        frag_weights = clients[3][0].train()
        clients[3][1].sendFragmentedWeightsToClients(frag_weights)

        nodes[1][0].cluster_weights = []
        for k, v in nodes[1][0].cluster.items():
            nodes[1][0].cluster[k] = 0

        clients[2][1].sendWeightsToNode(clients[2][0].sum_weights, 10)
        clients[3][1].sendWeightsToNode(clients[3][0].sum_weights, 10)

        weights = nodes[1][0].clusterAggregation()

        updateAndBroadcastModel(nodes[1], weights)

        print("done node 2")

        frag_weights = clients[4][0].train()
        clients[4][1].sendFragmentedWeightsToClients(frag_weights)
        frag_weights = clients[5][0].train()
        clients[5][1].sendFragmentedWeightsToClients(frag_weights)

        nodes[2][0].cluster_weights = []
        for k, v in nodes[2][0].cluster.items():
            nodes[2][0].cluster[k] = 0

        clients[4][1].sendWeightsToNode(clients[4][0].sum_weights, 10)
        clients[5][1].sendWeightsToNode(clients[5][0].sum_weights, 10)

        weights = nodes[2][0].clusterAggregation()

        updateAndBroadcastModel(nodes[2], weights)

        print("done node 3")

        answer = nodes[0][0].aggregateParameters(2)

        if answer["status"] == True:
            msg = nodes[0][1].broadcast_block(answer["message"]) 

            nodes[0][0].AppendBlock(answer["message"])
            nodes[0][1].broadcast_decision()

            broadcastModelToClients(nodes[0])

        else: 
            print(answer["message"]) 
            break

    saveNodesChain(nodes)




