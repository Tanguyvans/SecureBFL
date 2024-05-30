import time
import threading
import logging

from node import Node
from client import Client, data_preparation, get_dataset, create_sequences

import warnings
warnings.filterwarnings("ignore")

batch_size = 256


# %%
def create_nodes(train_sets, test_sets, number_of_nodes, dp=True, ss_type="additif", m=3,
                 name_dataset="Airline Satisfaction", model_choice="simplenet"):
    nodes = []

    for i in range(number_of_nodes): 
        nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=6010 + i,
                consensus_protocol="pbft",
                batch_size=batch_size,
                train=train_sets[0],
                test=test_sets[0],
                dp=dp,
                ss_type=ss_type,
                m=m,
                name_dataset=name_dataset,
                model_choice=model_choice
            )
        )
        
    return nodes


def create_clients(train_sets, test_sets, node, number_of_clients, dp=True, type_ss="additif", threshold=3, m=3,
                   name_dataset="Airline Satisfaction", model_choice="simplenet"):
    clients = {}
    for i in range(number_of_clients): 
        clients[f"c{node}_{i+1}"] = Client(
            id=f"c{node}_{i+1}",
            host="127.0.0.1",
            port=5010 + i + node * 10,
            batch_size=batch_size,
            train=train_sets[0],
            test=test_sets[0],
            dp=dp,
            type_ss=type_ss,
            threshold=threshold,
            m=m,
            name_dataset=name_dataset,
            model_choice=model_choice
        )

    return clients


def cluster_generation(nodes, clients, min_number_of_clients_in_cluster): 
    for i in range(numberOfNodes): 
        nodes[i].generate_clusters(min_number_of_clients_in_cluster)
        for cluster in nodes[i].clusters: 
            for client_id_1 in cluster.keys(): 
                if client_id_1 == "tot" or client_id_1 == "count":
                    continue
                clients[i][client_id_1].reset_connections()
                for client_id_2 in cluster.keys(): 
                    if client_id_2 == "tot" or client_id_2 == "count":
                        continue
                    if client_id_1 != client_id_2: 
                        clients[i][client_id_1].add_connections(client_id_2, clients[i][client_id_2].port)


# todo:
#  Gérer l'attente : attendre de recevoir les parts de k clients pour un cluster pour commencer shamir (pour le moment on attend les min_number_of_clients_in_cluster shares)
#  Ajouter le checksum pour verifier la non altération des shares du smpc.
# %%
if __name__ == "__main__":
    # %%
    logging.basicConfig(level=logging.DEBUG)
    # %%
    data_root = "Data"
    name_dataset = "Energy"  # "Airline Satisfaction" or "Energy"
    data_folder = f"{data_root}/{name_dataset}"
    ID = 1239
    numberOfNodes = 3
    numberOfClientsPerNode = 6  # corresponds to the number of clients per node, n in the shamir scheme
    min_number_of_clients_in_cluster = 3

    poisonned_number = 0
    epochs = 30
    ts = 10
    dp = False  # True if you want to apply differential privacy

    type_ss = "additif"  # "shamir" or "additif"
    k = 3  # The threshold : The minimum number of parts to reconstruct the secret (so with a polynomial of order k-1)
    m = min_number_of_clients_in_cluster  # The number of parts used to reconstruct the secret (M <= K)

    if m < k:
        raise ValueError("the number of parts used to reconstruct the secret must be greater than the threshold (k)")

    # %%

    with open("output.txt", "w") as f:
        f.write("")

    # %%
    if name_dataset == "Airline Satisfaction":
        model_choice = "simplenet"
        train_path = f'{data_folder}/train.csv'
        test_path = f'{data_folder}/test.csv'
        df_train = get_dataset(train_path, name_dataset)
        df_test = get_dataset(test_path, name_dataset)

        client_train_sets = data_preparation(df_train, name_dataset, numberOfClientsPerNode * numberOfNodes)
        client_test_sets = data_preparation(df_test, name_dataset, numberOfClientsPerNode * numberOfNodes)

        node_train_sets = data_preparation(df_train, name_dataset, numberOfClientsPerNode * numberOfNodes)
        node_test_sets = data_preparation(df_test, name_dataset, numberOfClientsPerNode * numberOfNodes)

        del df_train, df_test

    elif name_dataset == "Energy":
        model_choice = "LSTM"
        file_path = f"{data_folder}/Electricity/residential_{ID}.pkl"

        df = get_dataset(file_path, name_dataset)
        window_size = 10  # number of data points used as input to predict the next data point
        data_train = create_sequences(df["2009-07-15": "2010-07-14"], window_size)
        data_test = create_sequences(df["2010-07-14": "2010-07-21"], window_size)

        client_train_sets = data_preparation(data_train, name_dataset, numberOfClientsPerNode * numberOfNodes)
        client_test_sets = data_preparation(data_test, name_dataset, numberOfClientsPerNode * numberOfNodes)

        node_train_sets = data_preparation(data_train, name_dataset, numberOfClientsPerNode * numberOfNodes)
        node_test_sets = data_preparation(data_test, name_dataset, numberOfClientsPerNode * numberOfNodes)

        del df, data_train, data_test
    else:
        raise ValueError("The dataset name is not correct")
    # %%
    for i in range(poisonned_number):
        client_train_sets[i][1] = client_train_sets[i][1].replace({0: 1, 1: 0})
        client_test_sets[i][1] = client_test_sets[i][1].replace({0: 1, 1: 0})

    # %%
    nodes = create_nodes(node_train_sets, node_test_sets, numberOfNodes, dp=dp, ss_type=type_ss, m=m,
                         name_dataset=name_dataset, model_choice=model_choice)

    # %%## client to node connections ###
    clients = []
    for i in range(numberOfNodes): 
        node_clients = create_clients(client_train_sets, client_test_sets, i, numberOfClientsPerNode,
                                      dp, type_ss, k, m=m, name_dataset=name_dataset, model_choice=model_choice)
        clients.append(node_clients)
        for client_id, client in node_clients.items(): 
            client.update_node_connection(nodes[i].id, nodes[i].port)

    # %% All nodes connections ###
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

    # %% run threads ###
    for i in range(numberOfNodes): 
        threading.Thread(target=nodes[i].start_server).start()
        for client in clients[i].values(): 
            threading.Thread(target=client.start_server).start()

    nodes[0].create_first_global_model_request()
    time.sleep(10)

    # %% training and SMPC
    for epoch in range(epochs): 
        print(f"### EPOCH {epoch + 1} ###")
        ### Creation of the clusters + client to client connections ###
        cluster_generation(nodes, clients, min_number_of_clients_in_cluster)

        ### training ###
        for i in range(numberOfNodes):
            print(f"Node {i + 1} : Training\n")
            for client in clients[i].values():
                frag_weights = client.train()  # returns the shares to be sent to the other clients donc liste de la forme [(x1, y1), (x2, y2), ...]
                client.send_frag_clients(frag_weights)
        
        time.sleep(ts)

        ### SMPC ###
        for i in range(numberOfNodes):
            print(f"Node {i + 1} : SMPC\n")
            for client in clients[i].values():
                client.send_frag_node()
                time.sleep(ts)

        ### global model creation
        nodes[0].create_global_model()

        time.sleep(ts)

    nodes[0].blockchain.print_blockchain()

    # %%
    for i in range(numberOfNodes):
        nodes[i].blockchain.save_chain_in_file(f"node{i + 1}.txt")

