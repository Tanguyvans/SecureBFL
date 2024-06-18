import time
import threading
import logging

import numpy as np

from node import Node
from client import Client

from going_modular.data_setup import load_dataset
import warnings
warnings.filterwarnings("ignore")


# %%
def create_nodes(test_sets, number_of_nodes, coef_usefull=1.2, dp=True, ss_type="additif", m=3,
                 name_dataset="Airline Satisfaction", model_choice="simplenet", choice_loss="cross_entropy",
                 batch_size=256, num_classes=10):
    nodes = []

    for i in range(number_of_nodes): 
        nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=6010 + i,
                consensus_protocol="pbft",
                batch_size=batch_size,
                test=test_sets[i],
                coef_usefull=coef_usefull,
                dp=dp,
                ss_type=ss_type,
                m=m,
                name_dataset=name_dataset,
                model_choice=model_choice,
                choice_loss=choice_loss,
                num_classes=num_classes
            )
        )
        
    return nodes


def create_clients(train_sets, test_sets, node, number_of_clients, dp=True, type_ss="additif", threshold=3, m=3,
                   name_dataset="Airline Satisfaction", model_choice="simplenet",
                   choice_loss="cross_entropy", batch_size=256, epochs=3, num_classes=10):
    clients = {}
    for i in range(number_of_clients): 
        clients[f"c{node}_{i+1}"] = Client(
            id=f"c{node}_{i+1}",
            host="127.0.0.1",
            port=5010 + i + node * 10,
            batch_size=batch_size,
            train=train_sets[i],
            test=test_sets[i],
            dp=dp,
            type_ss=type_ss,
            threshold=threshold,
            m=m,
            name_dataset=name_dataset,
            model_choice=model_choice,
            choice_loss=choice_loss,
            epochs=epochs,
            num_classes=num_classes,
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
    name_dataset = "cifar"  # "Airline Satisfaction" or "Energy" or "cifar" or "mnist" or "alzheimer"
    batch_size = 256
    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"  # à ajouter
    choice_scheduler = "StepLR"  # à ajouter

    # nodes
    numberOfNodes = 3
    coef_usefull = 1

    # clients
    numberOfClientsPerNode = 3  # corresponds to the number of clients per node, n in the shamir scheme
    min_number_of_clients_in_cluster = 3
    client_epochs = 3
    poisonned_number = 0
    epochs = 5
    ts = 30
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
    n_classes = None
    length = None
    if name_dataset == "Airline Satisfaction":
        model_choice = "simplenet"
        choice_loss = 'bce_with_logits'

    elif name_dataset == "Energy":
        model_choice = "LSTM"  # "LSTM" or "GRU"
        choice_loss = 'mse'

    elif name_dataset == "cifar":
        model_choice = "CNNCifar"

    elif name_dataset == "mnist":
        model_choice = "CNNMnist"

    elif name_dataset == "alzheimer":
        model_choice = "mobilenet"
        length = 32

    else:
        raise ValueError("The dataset name is not correct")

    client_train_sets, client_test_sets, node_test_sets, n_classes = load_dataset(length, name_dataset,
                                                                                  data_root, numberOfClientsPerNode,
                                                                                  numberOfNodes)

    # %%
    # Change the poisonning for cifar
    for i in range(poisonned_number):
        if name_dataset == "Airline Satisfaction":
            client_train_sets[i][1] = client_train_sets[i][1].replace({0: 1, 1: 0})
            client_test_sets[i][1] = client_test_sets[i][1].replace({0: 1, 1: 0})

        elif name_dataset in ["cifar", "mnist", "alzheimer"]:
            n = len(client_train_sets[i][1])
            client_train_sets[i][1] = np.random.randint(0, n_classes, size=n).tolist()
            n = len(client_test_sets[i][1])
            client_test_sets[i][1] = np.random.randint(0, n_classes, size=n).tolist()

        else:
            raise ValueError("The dataset name is not correct")

    # the nodes should not have a train dataset
    nodes = create_nodes(
        node_test_sets, numberOfNodes, dp=dp, ss_type=type_ss, m=m,
        name_dataset=name_dataset, model_choice=model_choice, choice_loss=choice_loss,
        batch_size=batch_size, num_classes=n_classes
    )

    # %%## client to node connections ###
    clients = []
    for i in range(numberOfNodes): 
        node_clients = create_clients(
            client_train_sets, client_test_sets, i, numberOfClientsPerNode,
            dp, type_ss, k, m=m, name_dataset=name_dataset, model_choice=model_choice,
            choice_loss=choice_loss, batch_size=batch_size, epochs=client_epochs,
            num_classes=n_classes
        )
        clients.append(node_clients)
        for client_id, client in node_clients.items(): 
            client.update_node_connection(nodes[i].id, nodes[i].port)

    # %% All nodes connections ###
    for i in range(numberOfNodes): 
        node_i = nodes[i]
        # ## node to node ###
        for j in range(numberOfNodes): 
            node_j = nodes[j]
            if i != j: 
                node_i.add_peer(peer_id=node_j.id, peer_address=("localhost", node_j.port))

        # ## node to client ###
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
                frag_weights = client.train()  # returns the shares to be sent to the other clients, so list of the form [(x1, y1), (x2, y2), ...].
                client.send_frag_clients(frag_weights)
        
            time.sleep(ts*2)

            print(f"Node {i + 1} : SMPC\n")
            for client in clients[i].values():
                client.send_frag_node()
                time.sleep(ts)

            time.sleep(ts*2)

        ### global model creation

        nodes[0].create_global_model()

        time.sleep(ts)

    nodes[0].blockchain.print_blockchain()

    # %%
    for i in range(numberOfNodes):
        nodes[i].blockchain.save_chain_in_file(f"node{i + 1}.txt")

