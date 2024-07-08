import time
import threading
import logging

import numpy as np

from node import Node
from client import Client

import pickle

from sklearn.model_selection import train_test_split

import socket
import threading

import os

import time

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from flowerclient import FlowerClient
import pickle

from flwr.server.strategy.aggregate import aggregate
from sklearn.model_selection import train_test_split


import socket

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from flowerclient import FlowerClient
from node import get_keys, start_server
from going_modular.security import sum_shares

from going_modular.data_setup import load_dataset
import warnings
warnings.filterwarnings("ignore")

def get_keys(private_key_path, public_key_path):
    os.makedirs("keys/", exist_ok=True)
    if os.path.exists(private_key_path) and os.path.exists(public_key_path):
        with open(private_key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

        with open(public_key_path, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

    else:
        # Generate new keys
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Save keys to files
        with open(private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(public_key_path, 'wb') as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )

    return private_key, public_key

def start_server(host, port, handle_message, num_node):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Node {num_node} listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_message, args=(client_socket,)).start()

def aggregate_weights(weight_lists):
    """
    Aggregates a list of weights by averaging them.
    
    Args:
    weight_lists (list of list of np.array): Each element is a list of np.array,
                                             where each list corresponds to a model's weights.
    
    Returns:
    list of np.array: Averaged weights.
    """
    # Initialize the sum of weights as zero arrays based on the shape of the first model's weights
    sum_weights = [np.zeros_like(weights) for weights in weight_lists[0]]

    # Sum all weights
    for weights in weight_lists:
        for i, weight in enumerate(weights):
            sum_weights[i] += weight

    # Divide by the number of weight sets to get the average
    num_models = len(weight_lists)
    average_weights = [weights / num_models for weights in sum_weights]

    return average_weights

class Client:
    def __init__(self, id, host, port, train, test, type_ss="additif", threshold=3, m=3, **kwargs):
        self.id = id
        self.host = host
        self.port = port

        self.epochs = kwargs['epochs']
        self.type_ss = type_ss
        self.threshold = threshold
        self.m = m
        self.list_shapes = None

        self.frag_weights = []
        self.sum_dataset_number = 0

        self.node = {}
        self.connections = {}

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"

        self.get_keys(private_key_path, public_key_path)

        x_train, y_train = train
        x_test, y_test = test

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=y_train if kwargs['name_dataset'] == "Airline Satisfaction" else None)

        self.flower_client = FlowerClient.client(
            x_train=x_train, 
            x_val=x_val, 
            x_test=x_test, 
            y_train=y_train, 
            y_val=y_val, 
            y_test=y_test,
            **kwargs
            )
        
        self.res = None

    def start_server(self):
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

        if message_type == "global_model":
            weights = pickle.loads(message.get("value"))
            self.flower_client.set_parameters(weights)

        client_socket.close()

    def train(self):
        old_params = self.flower_client.get_parameters({})
        res = old_params[:]
        
        res, metrics = self.flower_client.fit(res, self.id, {})
        test_metrics = self.flower_client.evaluate(res, {})

        with open(f"output.txt", "a") as f:
            f.write(f"client {self.id}: data:{metrics['len_train']} train: {metrics['len_train']} train: {metrics['train_loss']} {metrics['train_acc']} val: {metrics['val_loss']} {metrics['val_acc']} test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")

        return res

    def update_node_connection(self, id, address):
        with open(f"keys/{id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.node = {"address": address, "public_key": public_key}

    @property
    def sum_weights(self):
        return sum_shares(self.frag_weights, self.type_ss)

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

###############################################################

class Node:
    def __init__(self, id, host, port, test, coef_usefull=1.01, **kwargs):
        self.id = id
        self.host = host
        self.port = port
        self.coef_usefull = coef_usefull

        self.peers = {}
        self.clients = {}
        self.clusters = []
        self.cluster_weights = []

        self.global_params_directory = ""
        self.nb_updates = 0

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
            result = self.consensus_protocol.handle_message(message)

            if result == "added":
                block = self.blockchain.blocks[-1]
                model_type = block.model_type

                if model_type == "update":
                    nb_updates = 0
                    for block in self.blockchain.blocks[::-1]:
                        if block.model_type == "update":
                            nb_updates += 1
                        else:
                            break

        client_socket.close()

    def get_weights(self, len_dataset=10):
        params_list = []
        for block in self.blockchain.blocks[::-1]:
            if block.model_type == "update":
                loaded_weights_dict = np.load(block.storage_reference)
                loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict) - 1)]

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
        loaded_weights_dict = np.load(filename)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

        for k, v in self.clients.items():
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
        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 0
        
        filename = f"models/m0.npz"
        self.global_params_directory = filename
        os.makedirs("models/", exist_ok=True)
        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        self.broadcast_model_to_clients(filename)

    def create_global_model(self, weights, index): 
        aggregated_weights = aggregate(
                [(weights[i], 20) for i in range(len(weights))]
        )

        metrics = self.flower_client.evaluate(aggregated_weights, {})

        self.flower_client.set_parameters(aggregated_weights)
        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 0
        
        filename = f"models/m{index}.npz"
        self.global_params_directory = filename
        os.makedirs("models/", exist_ok=True)
        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        print(f"flower aggregation {metrics}")
        self.broadcast_model_to_clients(filename)

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

    def add_client(self, client_id, client_address):
        with open(f"keys/{client_id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.clients[client_id] = {"address": client_address, "public_key": public_key}

###############################################################

client_weights = []

def train_client(client):
    weights = client.train()  # Train the client
    client_weights.append(weights)
    #client.send_frag_clients(frag_weights)  # Send the shares to other clients
    training_barrier.wait()  # Wait here until all clients have trained


# %%
def create_nodes(test_sets, number_of_nodes, coef_usefull=1.2, dp=True,
                 name_dataset="Airline Satisfaction", model_choice="simplenet", batch_size=256, classes=(*range(10),),
                 choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None):
    nodes = []
    for i in range(number_of_nodes):
        nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=6010 + i,
                test=test_sets[i],
                coef_usefull=coef_usefull,
                batch_size=batch_size,
                dp=dp,
                name_dataset=name_dataset,
                model_choice=model_choice,
                classes=classes,
                choice_loss=choice_loss,
                choice_optimizer=choice_optimizer,
                choice_scheduler=choice_scheduler
            )
        )
        
    return nodes


def create_clients(train_sets, test_sets, node, number_of_clients, dp=True, type_ss="additif", threshold=3, m=3,
                   name_dataset="Airline Satisfaction", model_choice="simplenet",
                   batch_size=256, epochs=3, num_classes=10,
                   choice_loss="cross_entropy", learning_rate = 0.003, choice_optimizer="Adam", choice_scheduler=None):
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
            batch_size=batch_size,
            epochs=epochs,
            dp=dp,
            name_dataset=name_dataset,
            model_choice=model_choice,
            classes=classes,
            choice_loss=choice_loss,
            learning_rate=learning_rate,
            choice_optimizer=choice_optimizer,
            choice_scheduler=choice_scheduler
        )

    return clients

if __name__ == "__main__":
    # %%
    logging.basicConfig(level=logging.DEBUG)
    # %%
    data_root = "Data"
    name_dataset = "cifar"  # "Airline Satisfaction" or "Energy" or "cifar" or "mnist" or "alzheimer"
    batch_size = 32
    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = None #"StepLR"
    learning_rate = 0.001

    numberOfNodes = 1
    numberOfClientsPerNode = 18

    client_epochs = 5
    poisonned_number = 0
    n_rounds = 20
    ts = 60
    diff_privacy = False  # True if you want to apply differential privacy

    training_barrier = threading.Barrier(numberOfClientsPerNode)

    with open("output.txt", "w") as f:
        f.write("")

    # %%
    classes = ()
    length = None
    if name_dataset == "Airline Satisfaction":
        model_choice = "simplenet"
        choice_loss = 'bce_with_logits'

    elif name_dataset == "Energy":
        model_choice = "LSTM"  # "LSTM" or "GRU"
        choice_loss = 'mse'

    elif name_dataset == "cifar":
        model_choice = "CNNCifar"  # CNN, CNNCifar, mobilenet

    elif name_dataset == "mnist":
        model_choice = "CNNMnist"

    elif name_dataset == "alzheimer":
        model_choice = "mobilenet"
        length = 32

    else:
        raise ValueError("The dataset name is not correct")

    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(length, name_dataset,
                                                                                     data_root, numberOfClientsPerNode,
                                                                                     numberOfNodes)
    n_classes = len(list_classes)

    # the nodes should not have a train dataset
    node = create_nodes(
        node_test_sets, numberOfNodes, dp=diff_privacy,
        name_dataset=name_dataset, model_choice=model_choice, batch_size=batch_size, classes=list_classes,
        choice_loss=choice_loss, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler
    )[0]

    # %%## client to node connections ###

    node_clients = create_clients(
        client_train_sets, client_test_sets, 0, numberOfClientsPerNode,
        dp=diff_privacy, name_dataset=name_dataset, model_choice=model_choice, batch_size=batch_size,
        epochs=client_epochs, num_classes=n_classes,
        choice_loss=choice_loss, learning_rate=learning_rate, choice_optimizer=choice_optimizer, choice_scheduler=choice_scheduler
    )

    for client_id, client in node_clients.items(): 
        client.update_node_connection(node.id, node.port)

    # ## node to client ###
    for client_j in node_clients.values():
        node.add_client(client_id=client_j.id, client_address=("localhost", client_j.port))

    print("done with the connections")

    # %% run threads ###
    threading.Thread(target=node.start_server).start()
    for client in node_clients.values(): 
        threading.Thread(target=client.start_server).start()

    node.create_first_global_model()
    time.sleep(30)

    # %% training and SMPC
    for round_i in range(n_rounds):
        print(f"### ROUND {round_i + 1} ###")

        threads = []
        for client in node_clients.values():
            t = threading.Thread(target=train_client, args=(client,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        node.create_global_model(client_weights, round_i)

        time.sleep(60)
        client_weights = []


