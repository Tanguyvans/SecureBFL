import time
import threading
import logging

import numpy as np
import random
import json

from node import Node
from client import Client

from going_modular.utils import initialize_parameters
from going_modular.data_setup import load_dataset
import warnings

from config import settings

warnings.filterwarnings("ignore")

def train_client(client):
    frag_weights = client.train()  # Train the client
    client.send_frag_clients(frag_weights)  # Send the shares to other clients
    training_barrier.wait()  # Wait here until all clients have trained

def create_nodes(test_sets, number_of_nodes, save_results, coef_usefull=1.2, ss_type="additif", m=3, **kwargs):
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
                save_results=save_results,
                **kwargs
            )
        )
        
    return nodes

def create_clients(train_sets, test_sets, node, number_of_clients, save_results, type_ss="additif", threshold=3, m=3, **kwargs):
    clients = {}
    for i in range(number_of_clients):
        dataset_index = node * number_of_clients + i
        clients[f"c{node}_{i+1}"] = Client(
            id=f"c{node}_{i+1}",
            host="127.0.0.1",
            port=5010 + i + node * 10,
            train=train_sets[dataset_index],
            test=test_sets[dataset_index],
            type_ss=type_ss,
            threshold=threshold,
            m=m,
            save_results=save_results,
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
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    (data_root, name_dataset, model_choice, batch_size, choice_loss, choice_optimizer, choice_scheduler,
    learning_rate, step_size, gamma, patience, roc_path, matrix_path, save_results,
    numberOfNodes, coef_usefull, numberOfClientsPerNode, min_number_of_clients_in_cluster, n_epochs,
    n_rounds, poisonned_number, ts, diff_privacy, training_barrier, type_ss, k, m) = initialize_parameters(settings,"BFL")
    
    poisoning_type = "distribution" # order, distribution

    # save results
    json_dict = {
        'settings': settings
    }
    with open(save_results + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(save_results + "output.txt", "w") as f:
        f.write("")

    length = 32 if name_dataset == 'alzheimer' else None
    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(length, name_dataset,
                                                                                     data_root, numberOfClientsPerNode,                                                                              numberOfNodes)
    n_classes = len(list_classes)

    if poisoning_type == "order":
        for i in range(poisonned_number):
            n = len(client_train_sets[i][1])
            client_train_sets[i][1] = np.random.randint(0, n_classes, size=n).tolist()
    else: 
        clients_per_node = numberOfClientsPerNode
        poison_per_node = poisonned_number // numberOfNodes
        for node in range(numberOfNodes):
            for i in range(poison_per_node):
                client_index = node * clients_per_node + i
                n = len(client_train_sets[client_index][1])
                client_train_sets[client_index][1] = np.random.randint(0, n_classes, size=n).tolist()

    # the nodes should not have a train dataset
    nodes = create_nodes(
        node_test_sets, numberOfNodes, coef_usefull=coef_usefull, dp=diff_privacy, ss_type=type_ss, m=m,
        model_choice=model_choice, batch_size=batch_size, classes=list_classes,
        choice_loss=choice_loss, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler,
        save_results=save_results, matrix_path=matrix_path, roc_path=roc_path, 
    )

    ### client to node connections ###
    clients = []
    for i in range(numberOfNodes): 
        node_clients = create_clients(
            client_train_sets, client_test_sets, i, numberOfClientsPerNode, type_ss=type_ss, m=m,
            dp=diff_privacy, model_choice=model_choice,
            batch_size=batch_size, epochs=n_epochs, classes=list_classes, learning_rate=learning_rate,
            choice_loss=choice_loss, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler,
            step_size=step_size, gamma=gamma,
            save_results=save_results, matrix_path=matrix_path, roc_path=roc_path,
            patience=patience,
        )
        clients.append(node_clients)
        for client_id, client in node_clients.items(): 
            client.update_node_connection(nodes[i].id, nodes[i].port)

    # All nodes connections ###
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

    # run threads ###
    for i in range(numberOfNodes): 
        threading.Thread(target=nodes[i].start_server).start()
        for client in clients[i].values(): 
            threading.Thread(target=client.start_server).start()

    nodes[0].create_first_global_model_request()

    time.sleep(ts*3)

    # training and SMPC
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

    #
    for i in range(numberOfNodes):
        nodes[i].blockchain.save_chain_in_file(save_results + f"node{i + 1}.txt")
