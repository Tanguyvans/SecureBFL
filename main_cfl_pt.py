import logging
import numpy as np
import threading
import os
import pickle
import time
import socket
import json
import torch

from flwr.server.strategy.aggregate import aggregate
from sklearn.model_selection import train_test_split

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from flowerclient import FlowerClient
from node import get_keys, start_server
from going_modular.security import sum_shares

from going_modular.utils import initialize_parameters
from going_modular.data_setup import load_dataset
from going_modular.security import data_poisoning
from config import settings

import warnings
warnings.filterwarnings("ignore")

class Client:
    def __init__(self, id, host, port, train, test, save_results, **kwargs):
        self.id = id
        self.host = host
        self.port = port

        self.frag_weights = []

        self.node = {}
        self.connections = {}

        self.save_results = save_results

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"

        self.get_keys(private_key_path, public_key_path)

        x_train, y_train = train
        x_test, y_test = test

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=None)

        self.flower_client = FlowerClient.client(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            **kwargs
        )

        self.res = None  # Why ? Where is it used ? Only line added in the init compared to the original.

    def start_server(self):
        # same
        start_server(self.host, self.port, self.handle_message, self.id)

    def handle_message(self, client_socket):
        data_length_bytes = client_socket.recv(4)
        if not data_length_bytes:
            return  # No data received, possibly handle this case as an error or log it
        data_length = int.from_bytes(data_length_bytes, byteorder='big')

        # Now read exactly data_length bytes
        data = b''
        while len(data) < data_length:
            packet = client_socket.recv(data_length - len(data))
            if not packet:
                break  # Connection closed, handle this case if necessary
            data += packet

        if len(data) < data_length:
            # Log this situation as an error or handle it appropriately
            print("Data was truncated or connection was closed prematurely.")
            return

        message = pickle.loads(data)

        message_type = message.get("type")

        # No if message_type == "frag_weights" because no SMPC.
        if message_type == "global_model":
            print("received_value")
            weights = pickle.loads(message.get("value"))
            self.flower_client.set_parameters(weights)

        client_socket.close()

    def train(self):
        old_params = self.flower_client.get_parameters({})
        res = old_params[:]

        res, metrics = self.flower_client.fit(res, self.id, {})
        test_metrics = self.flower_client.evaluate(res, {'name': f'Client {self.id}'})
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(
                f"client {self.id}: data:{metrics['len_train']} "
                f"train: {metrics['len_train']} train: {metrics['train_loss']} {metrics['train_acc']} "
                f"val: {metrics['val_loss']} {metrics['val_acc']} "
                f"test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")
        # No apply_smpc() so we don't return encrypted_list but res directly (and no self.frag_weights)
        return res

    def update_node_connection(self, id, address):
        # same
        with open(f"keys/{id}_public_key.pem", 'rb') as fi:
            public_key = serialization.load_pem_public_key(
                fi.read(),
                backend=default_backend()
            )

        self.node = {"address": address, "public_key": public_key}

    @property
    def sum_weights(self):
        # same
        return sum_shares(self.frag_weights, "additif")

    def get_keys(self, private_key_path, public_key_path):
        # same
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

class Node:
    def __init__(self, id, host, port, test, save_results, check_usefulness, coef_useful=1.05, tolerance_ceil=0.06, **kwargs):
        self.id = id
        self.host = host
        self.port = port
        self.clients = {}
        self.global_params_directory = ""
        self.save_results = save_results
        self.check_usefulness = check_usefulness
        self.coef_useful = coef_useful
        self.tolerance_ceil = tolerance_ceil

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"
        self.get_keys(private_key_path, public_key_path)

        x_test, y_test = test

        self.flower_client = FlowerClient.node(
            x_test=x_test, 
            y_test=y_test,
            **kwargs
        )

    def start_server(self):
        start_server(self.host, self.port, self.handle_message, self.id)

    def handle_message(self, client_socket):
        # First, read the length of the data
        data_length_bytes = client_socket.recv(4)
        if not data_length_bytes:
            return  # No data received, possibly handle this case as an error or log it
        data_length = int.from_bytes(data_length_bytes, byteorder='big')

        # Now read exactly data_length bytes
        data = b''
        while len(data) < data_length:
            packet = client_socket.recv(data_length - len(data))
            if not packet:
                break  # Connection closed, handle this case if necessary
            data += packet

        if len(data) < data_length:
            # Log this situation as an error or handle it appropriately
            print("Data was truncated or connection was closed prematurely.")
            return

        message = pickle.loads(data)
        message_type = message.get("type")

        if message_type == "frag_weights":
            pass

        else:
            print("in else")

        client_socket.close()

    def get_weights(self, len_dataset=10):
        # same
        params_list = []
        for block in self.blockchain.blocks[::-1]:
            if block.model_type == "update":
                loaded_weights_dict = np.load(block.storage_reference)
                loaded_weights = [val for name, val in loaded_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]
                loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])
                params_list.append(loaded_weights)

            else:
                break

        if len(params_list) == 0:
            return None

        self.aggregated_params = aggregate(params_list)

        self.flower_client.set_parameters(self.aggregated_params)

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        return weights_dict

    def broadcast_model_to_clients(self, filename):
        # Load the model parameters from the .pt file
        loaded_weights = torch.load(filename)

        print("len(self.clients)", len(self.clients))
        for k, v in self.clients.items():
            print("sending to client", k)
            address = v.get('address')

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('127.0.0.1', address[1]))

            serialized_data = pickle.dumps(loaded_weights)
            message = {"type": "global_model", "value": serialized_data}

            serialized_message = pickle.dumps(message)

            # Send the length of the message first
            client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)

            # Close the socket after sending
            client_socket.close()

    def create_first_global_model(self):
        os.makedirs("models/CFL", exist_ok=True)

        global_model = self.flower_client.get_parameters({})
        filename = f"models/CFL/m0.pt"
        self.global_params_directory = filename
        torch.save(global_model, filename)

        self.broadcast_model_to_clients(filename)

    def create_global_model(self, weights, index, two_step=False):
        useful_weights = []
        
        for i, client_weights in enumerate(weights):
            if self.check_usefulness:
                client_metrics = self.flower_client.evaluate(client_weights, {'name': f'Client model {i}'})
                current_global_metrics = self.flower_client.evaluate(self.flower_client.get_parameters({}), {'name': 'Current global model'})
            
                if self.is_update_useful(client_metrics, current_global_metrics):
                    useful_weights.append((client_weights, 10))  # Le 10 est un poids arbitraire, vous pouvez l'ajuster
                    print(f"Client model {i} is useful and will be included in aggregation")
                else:
                    print(f"Client model {i} is not useful and will be excluded from aggregation")

            else:
                useful_weights.append((client_weights, 10))

        if not useful_weights:
            print("No useful client models found. Keeping the current global model.")
            return

        # Agréger uniquement les modèles utiles
        aggregated_weights = aggregate(useful_weights)

        metrics = self.flower_client.evaluate(aggregated_weights, {})

        self.flower_client.set_parameters(aggregated_weights)

        global_model = self.flower_client.get_parameters({})
        filename = f"models/CFL/m{index}.pt"
        self.global_params_directory = filename
        torch.save(global_model, filename)

        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(f"Round {index}, Global aggregation: {metrics}\n")

        print(f"Round {index}, Global aggregation: {metrics}")
        self.broadcast_model_to_clients(filename)

    def is_update_useful(self, update_metrics, global_metrics):
        allowed_unimprovement = min(self.tolerance_ceil, global_metrics['test_loss'] * (self.coef_useful - 1))
        return update_metrics['test_loss'] <= global_metrics['test_loss'] + allowed_unimprovement

    def get_keys(self, private_key_path, public_key_path):
        # same
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

    def add_client(self, c_id, client_address):
        with open(f"keys/{c_id}_public_key.pem", 'rb') as fi:
            public_key = serialization.load_pem_public_key(
                fi.read(),
                backend=default_backend()
            )

        self.clients[c_id] = {"address": client_address, "public_key": public_key}
        print(self.clients)

def train_client(client_obj):
    weights = client_obj.train()  # Train the client
    client_weights.append(weights)
    training_barrier.wait()  # Wait here until all clients have trained

def create_nodes(test_sets, number_of_nodes, save_results, check_usefulness, coef_useful, tolerance_ceil, **kwargs):
    list_nodes = []
    for i in range(number_of_nodes):
        list_nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=6010 + i,
                test=test_sets[i],
                save_results=save_results,
                check_usefulness=check_usefulness,
                coef_useful=coef_useful,
                tolerance_ceil=tolerance_ceil,
                **kwargs
            )
        )

    return list_nodes

def create_clients(train_sets, test_sets, node, number_of_clients, save_results, **kwargs):
    # Exactly the same as the create_clients() function
    # but it called Client class from manual_cfl.py and not from client.py
    dict_clients = {}
    for i in range(number_of_clients):
        dict_clients[f"c{node}_{i + 1}"] = Client(
            id=f"c{node}_{i + 1}",
            host="127.0.0.1",
            port=5010 + i + node * 10,
            train=train_sets[i],
            test=test_sets[i],
            save_results=save_results,
            **kwargs
        )

    return dict_clients

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    training_barrier, length = initialize_parameters(settings, 'CFL')
    json_dict = {
        'settings': settings
    }

    with open(settings['save_results'] + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(settings['save_results'] + "output.txt", "w") as f:
        f.write("")

    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(length, settings['name_dataset'],
                                                                                     settings['data_root'],
                                                                                     settings['n_clients'],
                                                                                     1)

    data_poisoning(
        client_train_sets,
        poisoning_type="order",
        n_classes=len(list_classes),
        poisoned_number=settings['poisoned_number'],
    )

    # Create the server entity
    # the nodes should not have a train dataset
    server = create_nodes(
        node_test_sets, 1, 
        save_results=settings['save_results'],
        check_usefulness=settings['check_usefulness'],
        coef_useful=settings['coef_useful'],
        tolerance_ceil=settings['tolerance_ceil'],
        dp=settings['diff_privacy'], 
        model_choice=settings['arch'], 
        batch_size=settings['batch_size'],
        classes=list_classes, 
        choice_loss=settings['choice_loss'], 
        choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'], 
        save_figure=None, 
        matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'], pretrained=settings['pretrained'],
        save_model=settings['save_model']
    )[0]

    # ## client to node (central serveur) connections ###

    node_clients = create_clients(
        client_train_sets, client_test_sets, 0, settings['n_clients'], save_results=settings['save_results'],
        dp=settings['diff_privacy'], model_choice=settings['arch'], batch_size=settings['batch_size'],
        epochs=settings['n_epochs'], classes=list_classes, learning_rate=settings['lr'],
        choice_loss=settings['choice_loss'], choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'], step_size=settings['step_size'], gamma=settings['gamma'],
        save_figure=None, matrix_path=settings['matrix_path'], roc_path=settings['roc_path'],
        patience=settings['patience'], pretrained=settings['pretrained'],
        save_model=settings['save_model']
    )

    for client_id, client in node_clients.items():
        client.update_node_connection(server.id, server.port)

    # ## node to client ###
    for client_j in node_clients.values():
        server.add_client(c_id=client_j.id, client_address=("localhost", client_j.port))

    print("done with the connections")

    # ## run threads ###
    threading.Thread(target=server.start_server).start()
    for client in node_clients.values():
        threading.Thread(target=client.start_server).start()

    server.create_first_global_model()
    time.sleep(10)

    client_weights = []

    # ## training and SMPC
    for round_i in range(settings['n_rounds']):
        print(f"### ROUND {round_i + 1} ###")

        # Training
        threads = []
        for client in node_clients.values():
            t = threading.Thread(target=train_client, args=(client,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Agrégation avec vérification d'utilité
        server.create_global_model(client_weights, round_i + 1, two_step=False)

        time.sleep(settings['ts'])
        client_weights = []
