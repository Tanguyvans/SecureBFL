import time
import threading
import logging

import numpy as np
import random
import json

from node import Node
from client import Client

from going_modular.data_setup import load_dataset
import warnings
warnings.filterwarnings("ignore")


def train_client(client):
    frag_weights = client.train()  # Train the client
    client.send_frag_clients(frag_weights)  # Send the shares to other clients
    training_barrier.wait()  # Wait here until all clients have trained


# %%
def create_nodes(test_sets, number_of_nodes, coef_usefull=1.2, ss_type="additif", m=3, **kwargs):
    nodes = []
    for i in range(number_of_nodes):
        nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=6010 + i,
                consensus_protocol="pbft",
                test=test_sets[i],
                coef_usefull=coef_usefull,
                ss_type=ss_type,
                m=m,
                **kwargs
            )
        )
        
    return nodes


def create_clients(train_sets, test_sets, node, number_of_clients, type_ss="additif", threshold=3, m=3, **kwargs):
    clients = {}
    for i in range(number_of_clients):
        clients[f"c{node}_{i+1}"] = Client(
            id=f"c{node}_{i+1}",
            host="127.0.0.1",
            port=5010 + i + node * 10,
            train=train_sets[i],
            test=test_sets[i],
            type_ss=type_ss,
            threshold=threshold,
            m=m,
            **kwargs
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
#  Ajouter les courbes d'entrainement et de validation coté clients et Nodes car pour le moment problèmes de threads.
#  Utiliser la moyenne pondérée par la taille des datasets pour la reconstruction du modèle global (pour le moment on fait la moyenne arithmétique car même pondération pour tous les clients). Cela permettra de donner plus de poids aux modèles réalisées par des clients plus importants.
# %%
if __name__ == "__main__":
    # %%
    logging.basicConfig(level=logging.DEBUG)
    # %%
    data_root = "Data"
    name_dataset = "alzheimer"  # "cifar" or "mnist" or "alzheimer"
    model_choice = "simpleNet"  # "simpleNet" or "CNNCifar" or "CNNMnist" or "mobilenet"
    batch_size = 32
    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = "StepLR"  # 'cycliclr'
    learning_rate = 0.001
    step_size = 5
    gamma = 0.5
    patience = 3
    roc_path = None  # "roc"
    matrix_path = None  # "matrix"
    save_results = "results/BFL/"
    output_path = "results.txt"

    # nodes
    numberOfNodes = 3  # minimum 3 for the pbft (else they will not reach a consensus)
    coef_usefull = 2

    # clients
    numberOfClientsPerNode = 6  # corresponds to the number of clients per node, n in the shamir scheme
    min_number_of_clients_in_cluster = 3
    n_epochs = 20
    n_rounds = 25
    poisonned_number = 0
    ts = 10
    diff_privacy = False  # True if you want to apply differential privacy

    training_barrier = threading.Barrier(numberOfClientsPerNode)

    type_ss = "additif"  # "shamir" or "additif"
    k = 3  # The threshold : The minimum number of parts to reconstruct the secret (so with a polynomial of order k-1)
    m = min_number_of_clients_in_cluster  # The number of parts used to reconstruct the secret (M <= K)

    if m < k:
        raise ValueError("the number of parts used to reconstruct the secret must be greater than the threshold (k)")
    print("Number of Nodes: ", numberOfNodes,
          "\tNumber of Clients per Node: ", numberOfClientsPerNode,
          "\tNumber of Clients per Cluster: ", min_number_of_clients_in_cluster, "\n")

    # %% save results
    json_dict = {
        'settings': {
            'arch': model_choice,
            'pretrained': True,
            'name_dataset': name_dataset,
            'patience': patience,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            "numberOfNodes": numberOfNodes,
            "numberOfClientsPerNode": numberOfClientsPerNode,
            "min_number_of_clients_in_cluster": min_number_of_clients_in_cluster,
            'coef_usefull': coef_usefull,
            'poisonned_number': poisonned_number,
            "n_rounds": n_rounds,
            'choice_loss': choice_loss,
            'choice_optimizer': choice_optimizer,
            'lr': learning_rate,
            'choice_scheduler': choice_scheduler,
            'step_size': step_size,
            'gamma': gamma,
            'diff_privacy': diff_privacy,
            'secret_sharing': type_ss,
            'k': k,
            'm': m
        }
    }
    with open(save_results + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(save_results + output_path, "w") as f:
        f.write("")

    # %%
    length = 32 if name_dataset == 'alzheimer' else None
    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(length, name_dataset,
                                                                                     data_root, numberOfClientsPerNode,
                                                                                     numberOfNodes)
    n_classes = len(list_classes)

    # %%
    # Change the poisonning for cifar
    for i in random.sample(range(numberOfClientsPerNode * numberOfNodes), poisonned_number):
        print("Poisonning client ", i)
        n = len(client_train_sets[i][1])
        client_train_sets[i][1] = np.random.randint(0, n_classes, size=n).tolist()
        n = len(client_test_sets[i][1])
        client_test_sets[i][1] = np.random.randint(0, n_classes, size=n).tolist()

    # the nodes should not have a train dataset
    nodes = create_nodes(
        node_test_sets, numberOfNodes, coef_usefull=coef_usefull, dp=diff_privacy, ss_type=type_ss, m=m,
        model_choice=model_choice, batch_size=batch_size, classes=list_classes,
        choice_loss=choice_loss, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler,
        save_results=None, matrix_path=matrix_path, roc_path=roc_path,
    )

    # %%## client to node connections ###
    clients = []
    for i in range(numberOfNodes): 
        node_clients = create_clients(
            client_train_sets, client_test_sets, i, numberOfClientsPerNode, type_ss=type_ss, m=m,
            dp=diff_privacy, model_choice=model_choice,
            batch_size=batch_size, epochs=n_epochs, classes=list_classes, learning_rate=learning_rate,
            choice_loss=choice_loss, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler,
            step_size=step_size, gamma=gamma,
            save_results=None, matrix_path=matrix_path, roc_path=roc_path,
            patience=patience,
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

    time.sleep(ts*3)

    # %% training and SMPC
    for round_i in range(n_rounds):
        print(f"### ROUND {round_i + 1} ###")
        # ## Creation of the clusters + client to client connections ###
        cluster_generation(nodes, clients, min_number_of_clients_in_cluster)

        # ## training ###
        for i in range(numberOfNodes):
            print(f"Node {i + 1} : Training\n")
            threads = []
            for client in clients[i].values():
                t = threading.Thread(target=train_client, args=(client,))
                t.start()
                threads.append(t)
        
            for t in threads:
                t.join()

            print(f"Node {i + 1} : SMPC\n")
            
            for client in clients[i].values():
                client.send_frag_node()
                time.sleep(ts)

            time.sleep(ts*3)

        # ## global model creation

        nodes[0].create_global_model()

        time.sleep(ts)

    nodes[0].blockchain.print_blockchain()

    # %%
    for i in range(numberOfNodes):
        nodes[i].blockchain.save_chain_in_file(save_results + f"node{i + 1}.txt")
