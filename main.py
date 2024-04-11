import time
import threading
import logging

from node import Node
from client import Client, data_preparation

import warnings
warnings.filterwarnings("ignore")

def create_nodes(train_sets, test_sets, number_of_nodes, dp = True):
    nodes = []

    for i in range(number_of_nodes): 
        nodes.append(Node(id=f"n{i+1}", host="127.0.0.1", port=6010+i, consensus_protocol="pbft", batch_size=256, train=train_sets[0],
                test=test_sets[0], dp=dp))
        
    return nodes

def create_clients(train_sets, test_sets, node, number_of_clients, dp = True):
    clients = {}
    for i in range(number_of_clients): 
        clients[f"c{node}_{i+1}"] = Client(id=f"c{node}_{i+1}", host="127.0.0.1", port=5010+i+node*10, batch_size=256, train=train_sets[0], test=test_sets[0], dp=dp)

    return clients

def cluster_generation(nodes, clients, min_number_of_clients_in_cluster): 
    for i in range(numberOfNodes): 
        nodes[i].generate_clusters(min_number_of_clients_in_cluster)
        for cluster in nodes[i].clusters: 
            for client_id_1 in cluster.keys(): 
                if client_id_1 == "tot" or client_id_1 =="count": 
                    continue
                clients[i][client_id_1].reset_connections()
                for client_id_2 in cluster.keys(): 
                    if client_id_2 == "tot" or client_id_2 =="count": 
                        continue
                    if client_id_1 != client_id_2: 
                        clients[i][client_id_1].add_connections(client_id_2, clients[i][client_id_2].port)

if __name__ == "__main__": 
    logging.basicConfig(level=logging.DEBUG)

    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'

    numberOfNodes = 3
    numberOfClientsPerNode = 6
    min_number_of_clients_in_cluster = 3

    poisonned_number = 0
    epochs = 5
    ts = 10
    dp = True  # True if you want to apply differential privacy

    with open("output.txt", "w") as f:
        f.write("")

    client_train_sets = data_preparation(train_path, numberOfClientsPerNode*numberOfNodes)
    client_test_sets = data_preparation(test_path, numberOfClientsPerNode*numberOfNodes)

    node_train_sets = data_preparation(train_path, numberOfClientsPerNode*numberOfNodes)
    node_test_sets = data_preparation(test_path, numberOfClientsPerNode*numberOfNodes)

    for i in range(poisonned_number):
        client_train_sets[i][1] = client_train_sets[i][1].replace({0: 1, 1: 0})
        client_test_sets[i][1] = client_test_sets[i][1].replace({0: 1, 1: 0})

    nodes = create_nodes(node_train_sets, node_test_sets, numberOfNodes, dp = True)

    ### client to node connections ###
    clients = []
    for i in range(numberOfNodes): 
        node_clients = create_clients(client_train_sets, client_test_sets, i,numberOfClientsPerNode, dp)
        clients.append(node_clients)
        for client_id, client in node_clients.items(): 
            client.update_node_connection(nodes[i].id, nodes[i].port)
            # for client_id_2, client_2 in node_clients.items(): 
            #     if client_id != client_id_2: 
            #         client.add_connections(client_id_2, client_2.port)

    ### All nodes connections ###
    for i in range(numberOfNodes): 
        node_i = nodes[i]
        ### node to node ###
        for j in range(numberOfNodes): 
            node_j = nodes[j]
            if i != j: 
                node_i.add_peer(peer_id=node_j.id, peer_address=("localhost", node_j.port))
        ### node to client ###
        for client_j in clients[i].values(): 
            node_i.add_client(client_id=client_j.id, client_address=("localhost", client_j.port))

    ### run threads ###
    for i in range(numberOfNodes): 
        threading.Thread(target=nodes[i].start_server).start()
        for client in clients[i].values(): 
            threading.Thread(target=client.start_server).start()

    nodes[0].create_first_global_model_request()

    time.sleep(10)

    for epoch in range(epochs): 
        ### Creation of the clusters + client to client connections ###
        cluster_generation(nodes, clients, min_number_of_clients_in_cluster)

        ### training ###
        for i in range(numberOfNodes):
            for client in clients[i].values():
                frag_weights = client.train()
                client.send_frag_clients(frag_weights)

        time.sleep(ts)

        ### SMPC ###
        for i in range(numberOfNodes):
            for client in clients[i].values():
                client.send_frag_node()
                time.sleep(ts)

        ### global model creation
        nodes[0].create_global_model()
        time.sleep(ts)
        for i in range(1, numberOfNodes):
            nodes[i].global_params_directory = nodes[0].global_params_directory

        time.sleep(ts)


    # nodes[0].blockchain.print_blockchain()

    for i in range(numberOfNodes):
        nodes[i].blockchain.save_chain_in_file(f"node{i+1}.txt")
