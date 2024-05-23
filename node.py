import socket
import threading
import json
import os
import base64
from math import floor, ceil
import random

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

import hashlib
import numpy as np
from blockchain import Blockchain
from flowerclient import FlowerClient
import pickle

from flwr.server.strategy.aggregate import aggregate
from sklearn.model_selection import train_test_split

from protocols.pbft_protocol import PBFTProtocol
from protocols.raft_protocol import RaftProtocol


#%% Functions to reconstruct the shared secret on the node side with Shamir secret sharing
def generate_secret_shamir(x, y, m):
    """
    Function to generate the secret from the given points
    :param x: list of x
    :param y: list of y
    :param m: number of points to use for the reconstruction

    :return: the secret
    """
    # Initialisation of the answer
    ans = 0

    # loop to go through the given points
    for i in range(m):
        l = y[i]
        for j in range(m):
            # Compute the Lagrange polynomial
            if i != j:
                temp = -x[j] / (x[i] - x[j])  # L_i(x=0)
                l *= temp

        ans += l

    # return the secret
    return ans


def combine_shares_node(secret_list):
    """
    Function to combine the shares of the secret and get a dictionary with the good format for the decryption
    :param secret_list: list of shares of each client, so secret_list[id_client][x][layer]
    :return: dictionary of the secret, so secret_dic_final[x][layer]
    """
    secret_dic_final = {}
    for id_client in range(len(secret_list)):
        for x, list_weights in secret_list[id_client].items():
            if x in secret_dic_final:
                for layer in range(len(list_weights)):
                    secret_dic_final[x][layer] += list_weights[layer]
            else:
                secret_dic_final[x] = list_weights
    return secret_dic_final


def decrypt_shamir_node(secret_dic_final, secret_shape, m):
    """
    Function to decrypt the secret on the node side with Shamir secret sharing
    :param secret_dic_final: dictionary of the secret, so secret_dic_final[x][layer]
    :param secret_shape: list of shapes of the layers
    :param m: number of shares to use for the reconstruction of the secret
    :return: list of the decrypted secret, so decrypted_result[layer]  = weights_layer
    """
    x_combine = list(secret_dic_final.keys())
    y_combine = list(secret_dic_final.values())
    decrypted_result = []
    for layer in range(len(y_combine[0])):
        list_x = []
        list_y = []
        for i in range(m):  # (len(x_combine)):
            y = y_combine[i][layer]
            x = np.ones(y.shape) * x_combine[i]
            list_x.append(x)
            list_y.append(y)

        all_x_layer = np.array(list_x).T
        all_y_layer = np.array(list_y).T

        decrypted_result.append(
            np.round(
                [generate_secret_shamir(all_x_layer[i], all_y_layer[i], m) for i in range(len(all_x_layer))],
                4).reshape(secret_shape[layer]) / len(x_combine)
        )

    return decrypted_result


def aggregate_shamir(secret_list, secret_shape, m):
    """

    :param secret_list: list of shares of each client, so secret_list[id_client][x][layer]
    :param secret_shape: list of shapes of the layers
    :param m: number of shares to use for the reconstruction of the secret
    :return: dictionary of the secret, so secret_dic_final[x] = [y1, y2, y3, y4] if we have 4 layers.
    where x is the value of the x coordinate, and y1, y2, y3, y4 are the values of the y coordinate for each layer
    """
    secret_dic_final = combine_shares_node(secret_list)
    return decrypt_shamir_node(secret_dic_final, secret_shape, m)


# Other functions to handle the communication between the nodes
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


class Node:
    def __init__(self, id, host, port, consensus_protocol, batch_size, train, test, coef_usefull=1.05,
                 dp=False, ss_type="additif", m=3,
                 name_dataset="Airline Satisfaction", model_choice="simplenet"):
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

        x_train, y_train = train
        x_test, y_test = test
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=y_train if name_dataset == "Airline Satisfaction" else None)
        self.flower_client = FlowerClient(batch_size, x_train, x_val, x_test, y_train, y_val, y_test,
                                          dp, delta=1/(2*len(x_train)),
                                          name_dataset=name_dataset, model_choice=model_choice)
        self.ss_type = ss_type
        self.secret_shape = None
        self.m = m

        self.blockchain = Blockchain()
        if consensus_protocol == "pbft":
            self.consensus_protocol = PBFTProtocol(node=self, blockchain=self.blockchain)

        elif consensus_protocol == "raft":
            self.consensus_protocol = RaftProtocol(node=self, blockchain=self.blockchain)
            threading.Thread(target=self.consensus_protocol.run).start()

    def start_server(self):
        start_server(self.host, self.port, self.handle_message, self.id)

    def handle_message(self, client_socket):

        data_length = int.from_bytes(client_socket.recv(4), byteorder='big')
        data = client_socket.recv(data_length)
        message = pickle.loads(data)

        message_type = message.get("type")

        if message_type == "frag_weights":
            message_id = message.get("id")
            weights = pickle.loads(message.get("value"))
            self.secret_shape = message.get("list_shapes")

            for pos, cluster in enumerate(self.clusters):
                if message_id in cluster:
                    if cluster[message_id] == 0:
                        self.cluster_weights[pos].append(weights)

                        cluster[message_id] = 1
                        cluster["count"] += 1

                    if cluster["count"] == cluster["tot"]:
                        aggregated_weights = self.aggregation_cluster(pos)

                        participants = [k for k in cluster.keys() if k not in ["count", "tot"]]

                        message = self.create_update_request(aggregated_weights, participants)

                        self.consensus_protocol.handle_message(message)

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

                elif model_type == "global_model":

                    print(f"updating GM {self.global_params_directory}")
                    self.broadcast_model_to_clients()

        client_socket.close()

    def is_update_usefull(self, model_directory): 

        print(f"node: {self.id} GM: {self.global_params_directory}, {model_directory} ")
        if self.evaluate_model(model_directory)[0] <= self.evaluate_model(self.global_params_directory)[0]*self.coef_usefull:
            return True

        else: 
            return False

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

        self.aggregated_params = aggregate(params_list)

        self.flower_client.set_parameters(self.aggregated_params)

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        return weights_dict

    def is_global_valid(self, proposed_hash):
        weights_dict = self.get_weights(len_dataset=10)

        filename = f"models/{self.id}temp.npz"

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        if proposed_hash == self.calculate_model_hash(filename): 
            return True

        else:
            return False

    def evaluate_model(self, model_directory):
        loaded_weights_dict = np.load(model_directory)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loss = self.flower_client.evaluate(loaded_weights, {})[0]
        acc = self.flower_client.evaluate(loaded_weights, {})[2]['accuracy']

        return loss, acc

    def broadcast_model_to_clients(self):
        for block in self.blockchain.blocks[::-1]: 
            if block.model_type == "global_model":
                block_model = block 
                break 

        loaded_weights_dict = np.load(block_model.storage_reference)
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

    def broadcast_message(self, message):
        for peer_id in self.peers:
            self.send_message(peer_id, message)

    def send_message(self, peer_id, message):
        if peer_id in self.peers:
            peer_info = self.peers[peer_id]
            peer_address = peer_info["address"]

            signed_message = message.copy()
            signed_message["signature"] = self.sign_message(signed_message)
            signed_message["id"] = self.id

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect(peer_address)

                    serialized_message = pickle.dumps(signed_message)

                    # Send the length of the message first
                    client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
                    client_socket.send(serialized_message)

            except ConnectionRefusedError:
                pass

            except Exception as e:
                pass
        else:
            print(f"Peer {peer_id} not found.")

    def calculate_model_hash(self, filename): 
        loaded_weights_dict = np.load(filename)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])

        hash_model = hashlib.sha256()
        hash_model.update(str(loaded_weights).encode('utf-8'))
        hash_model = hash_model.hexdigest()

        return hash_model
   
    def create_first_global_model_request(self): 
        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        model_type = "global_model"
        
        filename = f"models/m0.npz"
        self.global_params_directory = filename
        os.makedirs("models/", exist_ok=True)
        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        message = {
            "id": self.id,
            "type": "request", 
            "content": {
                "storage_reference": filename,
                "model_type": model_type,
                "calculated_hash": self.calculate_model_hash(filename),
                "participants": ["1", "2"]
            }
        }

        self.consensus_protocol.handle_message(message)

    def create_global_model(self): 
        weights_dict = self.get_weights(len_dataset=10)

        model_type = "global_model"

        filename = f"models/{self.id}m{self.blockchain.len_chain}.npz"
        self.global_params_directory = filename

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        message = {
            "id": self.id,
            "type": "request", 
            "content": {
                "storage_reference": filename,
                "model_type": model_type,
                "calculated_hash": self.calculate_model_hash(filename), 
                "participants": ["1", "2"]
            }
        }

        self.consensus_protocol.handle_message(message)

    def create_update_request(self, weights, participants):

        self.flower_client.set_parameters(weights)

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 10
        model_type = "update"

        filename = f"models/m{self.blockchain.len_chain}.npz"

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        message = {
            "id": self.id,
            "type": "request", 
            "content": {
                "storage_reference": filename,
                "model_type": model_type,
                "calculated_hash": self.calculate_model_hash(filename),
                "participants": participants
            }
        }

        # self.isModelUpdateUsefull(filename)

        return message

    def aggregation_cluster(self, pos):
        if self.ss_type == "additif":
            aggregated_weights = aggregate(
                [(self.cluster_weights[pos][i], 20) for i in range(len(self.cluster_weights[pos]))]
            )
        else:
            # shamir secret sharing
            aggregated_weights = aggregate_shamir(self.cluster_weights[pos], self.secret_shape, self.m)

        loss = self.flower_client.evaluate(aggregated_weights, {})[0]

        with open('output.txt', 'a') as f: 
            f.write(f"cluster {pos} node {self.id}: {loss} \n")

        self.cluster_weights[pos] = []
        for k, v in self.clusters[pos].items():
            if k != "tot":
                self.clusters[pos][k] = 0

        return aggregated_weights

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

    def sign_message(self, message):
        signature = self.private_key.sign(
            json.dumps(message).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def verify_signature(self, signature, message, public_key):
        try:
            signature_binary = base64.b64decode(signature)

            public_key.verify(
                signature_binary,
                json.dumps(message).encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            print(f"Signature verification error: {e}")
            return False

    def add_peer(self, peer_id, peer_address):
        with open(f"keys/{peer_id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.peers[peer_id] = {"address": peer_address, "public_key": public_key}

    def add_client(self, client_id, client_address):
        with open(f"keys/{client_id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

        self.clients[client_id] = {"address": client_address, "public_key": public_key}

    def generate_clusters(self, min_number_of_clients): 
        self.clusters = []
        clients = [k for k in self.clients.keys()]
        random.shuffle(clients)

        number_of_clusters = floor(len(self.clients)/min_number_of_clients)
        max_number_of_clients = ceil(len(self.clients)/number_of_clusters)

        sol = [0]
        n = len(self.clients)
        for i in range(number_of_clusters): 
            if n % min_number_of_clients == 0: 
                sol.append(sol[-1] + min_number_of_clients)
                n -= min_number_of_clients
            else: 
                sol.append(sol[-1] + max_number_of_clients)
                n -= max_number_of_clients

        for i in range(1, len(sol)): 
            self.create_cluster(clients[sol[i-1]: sol[i]])

        print(self.clusters)

    def create_cluster(self, clients): 
        self.clusters.append({client: 0 for client in clients})
        self.clusters[-1]["tot"] = len(self.clusters[-1])
        self.clusters[-1]["count"] = 0

        self.cluster_weights.append([])
