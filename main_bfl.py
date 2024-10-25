import time
import threading
import logging
import json

from node import Node
from client import Client

from going_modular.utils import initialize_parameters
from going_modular.data_setup import load_dataset
from going_modular.security import data_poisoning
import warnings

from config import settings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")


# %%
def train_client(client_obj):
    frag_weights = client_obj.train()  # Train the client
    client_obj.send_frag_clients(frag_weights)  # Send the shares to other clients
    training_barrier.wait()  # Wait here until all clients have trained


def create_nodes(test_sets, number_of_nodes, save_results, coef_usefull=1.2, tolerance_ceil=0.1, ss_type="additif", m=3, **kwargs):
    list_nodes = []
    for num_node in range(number_of_nodes):
        list_nodes.append(
            Node(
                id=f"n{num_node + 1}",
                host="127.0.0.1",
                port=6010 + num_node,
                consensus_protocol="pbft",
                test=test_sets[num_node],
                coef_usefull=coef_usefull,
                tolerance_ceil=tolerance_ceil,
                ss_type=ss_type,
                m=m,
                save_results=save_results,
                **kwargs
            )
        )
        
    return list_nodes


def create_clients(train_sets, test_sets, node, number_of_clients, save_results, type_ss="additif", threshold=3, m=3,
                   **kwargs):
    dict_clients = {}
    for num_client in range(number_of_clients):
        dataset_index = node * number_of_clients + num_client
        dict_clients[f"c{node}_{num_client + 1}"] = Client(
            id=f"c{node}_{num_client + 1}",
            host="127.0.0.1",
            port=5010 + num_client + node * 10,
            train=train_sets[dataset_index],
            test=test_sets[dataset_index],
            type_ss=type_ss,
            threshold=threshold,
            m=m,
            save_results=save_results,
            **kwargs
        )

    return dict_clients


def cluster_generation(list_nodes, list_clients, min_number_of_clients_in_cluster, number_of_nodes):
    for num_node in range(number_of_nodes):
        list_nodes[num_node].generate_clusters(min_number_of_clients_in_cluster)
        for cluster in list_nodes[num_node].clusters:
            for client_id_1 in cluster.keys(): 
                if client_id_1 == "tot" or client_id_1 == "count":
                    continue
                list_clients[num_node][client_id_1].reset_connections()
                for client_id_2 in cluster.keys(): 
                    if client_id_2 == "tot" or client_id_2 == "count":
                        continue
                    if client_id_1 != client_id_2: 
                        list_clients[num_node][client_id_1].add_connections(client_id_2, list_clients[num_node][client_id_2].port)


# todo:
#  Gérer l'attente : attendre de recevoir les parts de k clients pour un cluster pour commencer shamir (pour le moment on attend les min_number_of_clients_in_cluster shares)
#  Ajouter le checksum pour verifier la non altération des shares du smpc.
#  Ajouter les courbes d'entrainement et de validation coté clients et Nodes car pour le moment problèmes de threads.
#  Utiliser la moyenne pondérée par la taille des datasets pour la reconstruction du modèle global (pour le moment on fait la moyenne arithmétique car même pondération pour tous les clients). Cela permettra de donner plus de poids aux modèles réalisées par des clients plus importants.
# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # %% Parameters
    training_barrier, length = initialize_parameters(settings, "BFL")

    poisoning_type = "distribution"  # order, distribution

    # %% save results
    json_dict = {
        'settings': {**settings, "length": length, "poisoning_type": poisoning_type}
    }
    with open(settings['save_results'] + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(settings['save_results'] + 'output.txt', "w") as f:
        f.write("")

    # Load dataset
    (client_train_sets, client_test_sets, node_test_sets, list_classes) = load_dataset(length, settings['name_dataset'],
                                                                                       settings['data_root'],
                                                                                       settings['number_of_clients_per_node'],
                                                                                       settings['number_of_nodes'])

    # Data poisoning of the clients
    data_poisoning(
        data=client_train_sets,
        poisoning_type=poisoning_type,
        n_classes=len(list_classes),
        poisoned_number=settings['poisoned_number'],
        number_of_nodes=settings['number_of_nodes'],
        clients_per_node=settings['number_of_clients_per_node'],
    )

    # Create nodes
    # the nodes should not have a train dataset
    nodes = create_nodes(
        node_test_sets, settings['number_of_nodes'], save_results=settings['save_results'],
        coef_usefull=settings['coef_usefull'], tolerance_ceil=settings['tolerance_ceil'], 
        ss_type=settings['secret_sharing'], m=settings['m'],
        dp=settings['diff_privacy'], model_choice=settings['arch'], batch_size=settings['batch_size'],
        classes=list_classes, choice_loss=settings['choice_loss'], choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'],  save_figure=None, matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'], pretrained=settings['pretrained'],
        save_model=settings['save_model']
    )

    # ## client to node connections ###
    clients = []
    for i in range(settings['number_of_nodes']):
        node_clients = create_clients(
            client_train_sets, client_test_sets, i, settings['number_of_clients_per_node'],
            type_ss=settings['secret_sharing'], m=settings['m'], threshold=settings['k'],
            save_results=settings['save_results'], dp=settings['diff_privacy'], model_choice=settings['arch'],
            batch_size=settings['batch_size'], epochs=settings['n_epochs'], classes=list_classes,
            learning_rate=settings['lr'], choice_loss=settings['choice_loss'],
            choice_optimizer=settings['choice_optimizer'], choice_scheduler=settings['choice_scheduler'],
            step_size=settings['step_size'], gamma=settings['gamma'], save_figure=None,
            matrix_path=settings['matrix_path'], roc_path=settings['roc_path'], patience=settings['patience'],
            pretrained=settings['pretrained'],
            save_model=settings['save_model']
        )
        clients.append(node_clients)
        for client_id, client in node_clients.items(): 
            client.update_node_connection(nodes[i].id, nodes[i].port)

    # All nodes connections ###
    for i in range(settings['number_of_nodes']):
        node_i = nodes[i]
        # ## node to node ###
        for j in range(settings['number_of_nodes']):
            node_j = nodes[j]
            if i != j: 
                node_i.add_peer(peer_id=node_j.id, peer_address=("localhost", node_j.port))

        # ## node to client ###
        for client_j in clients[i].values():
            node_i.add_client(client_id=client_j.id, client_address=("localhost", client_j.port))

    # run threads ###
    for i in range(settings['number_of_nodes']):
        threading.Thread(target=nodes[i].start_server).start()
        for client in clients[i].values(): 
            threading.Thread(target=client.start_server).start()

    nodes[0].create_first_global_model_request()

    time.sleep(settings['ts'])

    # training and SMPC
    for round_i in range(settings['n_rounds']):
        print(f"### ROUND {round_i + 1} ###")
        # ## Creation of the clusters + client to client connections ###
        cluster_generation(nodes, clients, settings['min_number_of_clients_in_cluster'], settings['number_of_nodes'])

        with open("results/BFL/output.txt", "a") as f:
            f.write(f"### ROUND {round_i + 1} ###\n")

        # ## training ###
        for i in range(settings['number_of_nodes']):
            print(f"Node {i + 1} : Training\n")
            threads = []
            for client in clients[i].values():
                t = threading.Thread(target=train_client, args=(client,))
                t.start()
                threads.append(t)
        
            for t in threads:
                t.join()

            print(f"Node {i + 1} : SMPC\n")
            
            for cluster in nodes[i].clusters:
                for client_id in cluster.keys():
                    if client_id in ['tot', 'count']:
                        continue
                    client = clients[i][client_id]
                    client.send_frag_node()
                    time.sleep(5)

                time.sleep(settings['ts'])

            time.sleep(settings['ts'])

        nodes[0].create_global_model()

        time.sleep(settings['ts'])

    nodes[0].blockchain.print_blockchain()

    for i in range(settings['number_of_nodes']):
        nodes[i].blockchain.save_chain_in_file(settings['save_results'] + f"node{i + 1}.txt")

    print("This is the end")
