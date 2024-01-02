import pickle
import random
import concurrent.futures
from sklearn.model_selection import train_test_split
import numpy as np
import json
import pandas as pd
import os
import time
import logging

import socket
import threading

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from flowerclient import FlowerClient

from node import Node

def encrypt(x, n_shares=2):
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)

# Fonction pour appliquer le SMPC à une liste de listes
def apply_smpc(input_list, n_shares=2):
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
# Fonction pour décrypter une liste de listes chiffrées
def decrypt_list_of_lists(encrypted_list):
    decrypted_list = []
    n_shares = len(encrypted_list)

    for i in range(len(encrypted_list[0])): 
        sum_array = np.add(encrypted_list[0][i], encrypted_list[1][i])
        for j in range(2, n_shares): 
            sum_array = np.add(sum_array, encrypted_list[j][i])

        decrypted_list.append(sum_array)

    return decrypted_list

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

        self.getKeys(private_key_path, public_key_path)

        X_train, y_train = train
        X_test, y_test = test
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

        self.flower_client = FlowerClient(batch_size, X_train, X_val ,X_test, y_train, y_val, y_test)

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Node {self.id} listening on {self.host}:{self.port}")

        while True:
            client_socket, addr = server_socket.accept()
            threading.Thread(target=self.handle_message, args=(client_socket,)).start()
    
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
        for i in range(2):
            res = self.flower_client.fit(res, {})[0]
            loss = self.flower_client.evaluate(res, {})[0]
            with open('output.txt', 'a') as f: 
                f.write(f"loss: {loss} \n")

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

            # Fermer le socket après l'envoi
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

        # Fermer le socket après l'envoi
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

    def getKeys(self, private_key_path, public_key_path): 
        if os.path.exists(private_key_path) and os.path.exists(public_key_path):
            with open(private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )

            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
        else:
            # Generate new keys
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()

            # Save keys to files
            with open(private_key_path, 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            with open(public_key_path, 'wb') as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))

if __name__ == "__main__": 
    # logging.basicConfig(level=logging.DEBUG)

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

            X_s = subset.drop(['satisfaction'],axis=1)
            y_s=subset[['satisfaction']]

            multi_df.append([X_s, y_s])

        return multi_df

    def saveNodesChain(nodes): 
        for node in nodes: 
            node[0].blockchain.save_chain_in_file(node[0].id)


    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'

    numberOfClients = 3
    poisonned_number = 0
    epochs = 20

    with open("output.txt", "w") as f: 
        f.write("")

    train_sets = dataPreparation(train_path, numberOfClients)
    test_sets = dataPreparation(test_path, numberOfClients)

    for i in range(poisonned_number): 
        train_sets[i][1] = train_sets[i][1].replace({0: 1, 1: 0})
        test_sets[i][1] = test_sets[i][1].replace({0: 1, 1: 0})

    node1 = Node(id="n1", host="127.0.0.1", port=6010, consensus_protocol="pbft", batch_size=256, train=train_sets[0], test=test_sets[0])
    node2 = Node(id="n2", host="127.0.0.1", port=6011, consensus_protocol="pbft", batch_size=256, train=train_sets[0], test=test_sets[0])
    node3 = Node(id="n3", host="127.0.0.1", port=6012, consensus_protocol="pbft", batch_size=256, train=train_sets[0], test=test_sets[0])

    client1 = Client(id="c1", host="127.0.0.1", port=5010, batch_size=256, train=train_sets[0], test=test_sets[0])
    client2 = Client(id="c2", host="127.0.0.1", port=5011, batch_size=256, train=train_sets[1], test=test_sets[1])
    client3 = Client(id="c3", host="127.0.0.1", port=5012, batch_size=256, train=train_sets[2], test=test_sets[2])

    client1.add_connections("c2", 5011)
    client1.add_connections("c3", 5012)

    client2.add_connections("c1", 5010)
    client2.add_connections("c3", 5012)

    client3.add_connections("c1", 5010)
    client3.add_connections("c2", 5011)

    client1.update_node_connection("n1", 6010)
    client2.update_node_connection("n1", 6010)
    client3.update_node_connection("n1", 6010)

    node1.add_peer(peer_id="n2", peer_address=("localhost", 6011))
    node1.add_peer(peer_id="n3", peer_address=("localhost", 6012))

    node2.add_peer(peer_id="n1", peer_address=("localhost", 6010))
    node2.add_peer(peer_id="n3", peer_address=("localhost", 6012))

    node3.add_peer(peer_id="n1", peer_address=("localhost", 6010))
    node3.add_peer(peer_id="n2", peer_address=("localhost", 6011))

    node1.add_client(client_id="c1", client_address=("localhost", 5010))
    node1.add_client(client_id="c2", client_address=("localhost", 5011))
    node1.add_client(client_id="c3", client_address=("localhost", 5012))

    node1.create_cluster([client1.id, client2.id, client3.id])

    threading.Thread(target=client1.start_server).start()
    threading.Thread(target=client2.start_server).start()
    threading.Thread(target=client3.start_server).start()

    threading.Thread(target=node1.start_server).start()
    threading.Thread(target=node2.start_server).start()
    threading.Thread(target=node3.start_server).start()

    node1.create_first_global_model_request()

    time.sleep(10)

    for i in range(5): 
        frag_weights_1 = client1.train()
        frag_weights_2 = client2.train()
        frag_weights_3 = client3.train()

        client1.send_frag_clients(frag_weights_1)
        client2.send_frag_clients(frag_weights_2)
        client3.send_frag_clients(frag_weights_3)

        client1.send_frag_node()
        client2.send_frag_node()
        client3.send_frag_node()

        time.sleep(5)

        if i > 1 and i % 2 == 0: 
            node1.create_global_model()

    time.sleep(5)

    node1.blockchain.print_blockchain()
    node2.blockchain.print_blockchain()
    node3.blockchain.print_blockchain()