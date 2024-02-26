import pickle
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import socket

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from flowerclient import FlowerClient
from node import get_keys, start_server


def encrypt(x, n_shares=2):
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)

def apply_smpc(input_list, n_shares=2):
    """
    Function to apply SMPC to a list of lists

    :param input_list: list of lists to encrypt
    :param n_shares: number of shares to create
    :return:
    """
    encrypted_list = [[] for i in range(n_shares)]
    for inner_list in input_list:
        if isinstance(inner_list[0], np.ndarray):
            encrypted_inner_list = [[] for i in range(n_shares)]
            for inner_inner_list in inner_list: 
                encrypted_inner_inner_list = [[] for i in range(n_shares)]
                for x in inner_inner_list: 
                    crypted_tuple = encrypt(x, n_shares)
                    for i in range(n_shares):
                        encrypted_inner_inner_list[i].append(crypted_tuple[i])

                for i in range(n_shares):
                    encrypted_inner_list[i].append(encrypted_inner_inner_list[i])
        else: 
            encrypted_inner_list = [[] for i in range(n_shares)]
            for x in inner_list: 
                crypted_tuple = encrypt(x, n_shares)

                for i in range(n_shares):
                    encrypted_inner_list[i].append(crypted_tuple[i])
        
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_inner_list[i]))

    return encrypted_list

def decrypt_list_of_lists(encrypted_list):
    """
    Function to decrypt a list of lists encrypted with SMPC
    :param encrypted_list:
    :return:
    """
    decrypted_list = []
    n_shares = len(encrypted_list)

    for i in range(len(encrypted_list[0])): 
        sum_array = np.add(encrypted_list[0][i], encrypted_list[1][i])
        for j in range(2, n_shares): 
            sum_array = np.add(sum_array, encrypted_list[j][i])

        decrypted_list.append(sum_array)

    return decrypted_list

def data_preparation(filename, number_of_nodes=3):
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
    df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
    df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    df = df.dropna()

    num_samples = len(df)

    split_size = num_samples // number_of_nodes

    multi_df = []
    for i in range(number_of_nodes):
        if i < number_of_nodes-1:
            subset = df.iloc[i*split_size:(i+1)*split_size]
        else:
            subset = df.iloc[i*split_size:]

        x_s = subset.drop(['satisfaction'], axis=1)
        y_s = subset[['satisfaction']]

        multi_df.append([x_s, y_s])

    return multi_df

def save_nodes_chain(nodes):
    for node in nodes:
        node[0].blockchain.save_chain_in_file(node[0].id)

class Client: 
    def __init__(self, id, host, port, batch_size, train, test):
        self.id = id
        self.host = host
        self.port = port
        self.cluster = 0

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
                                                          stratify=y_train)

        self.flower_client = FlowerClient(batch_size, x_train, x_val, x_test, y_train, y_val, y_test)

    def start_server(self):
        start_server(self.host, self.port, self.handle_message, self.id)
    
    def handle_message(self, client_socket):
        data_length = int.from_bytes(client_socket.recv(4), byteorder='big')
        data = client_socket.recv(data_length)
        
        message = pickle.loads(data)
        
        message_type = message.get("type")

        if message_type == "frag_weights": 
            weights = pickle.loads(message.get("value"))
            self.frag_weights.append(weights)
        elif message_type == "global_model": 
            weights = pickle.loads(message.get("value"))
            self.flower_client.set_parameters(weights)

        client_socket.close()

    def train(self): 
        old_params = self.flower_client.get_parameters({})
        res = old_params[:]
        for i in range(1):
            res = self.flower_client.fit(res, {})[0]
            loss = self.flower_client.evaluate(res, {})[0]
            with open('output.txt', 'a') as f: 
                f.write(f"client {self.id}: {loss} \n")

        encripted_lists = apply_smpc(res, len(self.connections)+1)
        self.frag_weights.append(encripted_lists.pop())

        return encripted_lists
    
    def send_frag_clients(self, frag_weights): 
        i = 0
        for k, v in self.connections.items(): 
            address = v.get('address')

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('127.0.0.1', address))

            serialized_data = pickle.dumps(frag_weights[i])
            message = {"type": "frag_weights", "value": serialized_data}

            serialized_message = pickle.dumps(message)

            # Send the length of the message first
            client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)

            # close the socket after sending
            client_socket.close()
            i += 1

    def send_frag_node(self): 
        address = self.node.get('address')

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', address))

        serialized_data = pickle.dumps(self.sum_weights)
        message = {"type": "frag_weights", "id": self.id, "value": serialized_data}

        serialized_message = pickle.dumps(message)

        # Send the length of the message first
        client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
        client_socket.send(serialized_message)

        # Close the socket after sending
        client_socket.close()

        self.frag_weights = []
    
    def add_connections(self, id, address):
        with open(f"keys/{id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.connections[id] = {"address": address, "public_key": public_key}

    def update_node_connection(self, id, address):
        with open(f"keys/{id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.node = {"address": address, "public_key": public_key}

    @property
    def sum_weights(self): 
        return decrypt_list_of_lists(self.frag_weights)

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)
