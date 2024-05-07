import pickle
# import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import socket

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from flowerclient import FlowerClient
from node import get_keys, start_server


# %% list of functions to apply secret sharing algorithm
def apply_smpc(input_list, n_shares=2, type_ss="additif", threshold=3):
    """
    Function to apply secret sharing algorithm and encode the given secret
    :param input_list: list of values to share
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :param type_ss: the type of secret sharing algorithm to use (by default additif)
    :param threshold: the minimum number of parts to reconstruct the secret (so with a polynomial of order threshold-1)
    :return: the list of shares for each client and the shape of the secret
    """
    if type_ss == "additif":
        return apply_additif(input_list, n_shares), None

    elif type_ss == "shamir":
        secret_shape = [weights_layer.shape for weights_layer in input_list]

        encrypted_result = apply_shamir(input_list, n_shares=n_shares, k=threshold)
        indices = [i for i in range(len(encrypted_result))]  # We get the values of x associated with each y.
        list_x = [i + 1 for i in range(len(encrypted_result))]
        # random.shuffle(indices)
        list_shares = [(list_x[id_client], encrypted_result[id_client])for id_client in indices]
        return list_shares, secret_shape

    else:
        raise ValueError("Type of secret sharing not recognized")


# Functions for additif secret sharing

def encrypt_tensor(secret, n_shares=3):
    """
    Function to encrypt a tensor of values with additif secret sharing
    :param secret: the tensor of values to encrypt
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :return: the list of shares for each client where each share is a tensor of values
    """
    shares = [np.random.randint(-5, 5, size=secret.shape) for _ in range(n_shares - 1)]

    # The last part is deduced from the sum of the previous parts to guarantee additivity
    shares.append(secret - sum(shares))

    return shares


def apply_additif(input_list, n_shares=3):
    """
    Function to apply secret sharing algorithm and encode the given secret when the secret is a tensor of values
    :param input_list: The secret to share (a tensor of values of type numpy.ndarray)
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :return: the list of shares for each client where each share is a list of tensors (one tensor for each layer)

    """
    encrypted_list = [[] for _ in range(n_shares)]
    for weights_layer in input_list:
        # List for the given layer where each element is a share for a client.
        encrypted_layer = encrypt_tensor(weights_layer, n_shares)  # crypter un tenseur de poids

        # For each layer, each client i has a part of each row of the given layer
        # because we add for client i, the encrypted_layer we prepared for this client
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_layer[i]))

    return encrypted_list


# Functions for shamir secret sharing
def calculate_y(x, poly):
    """
    Function to calculate the value of y from a polynomial and a value of x:
    y = poly[0] + x*poly[1] + x^2*poly[2] + ...

    :param x: the value of x
    :param poly: the list of coefficients of the polynomial

    :return: the value of y
    """
    y = sum([poly[i] * x ** i for i in range(len(poly))])
    return y


def apply_shamir(input_list, n_shares=2, k=3):
    """
    Function to apply shamir's secret sharing algorithm
    :param input_list: secret to share (a list of tensors of values of type numpy.ndarray)
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :param k: the minimum number of parts to reconstruct the secret, called threshold (so with a polynomial of order k-1)
    :return: the list of shares for each client
    """
    list_clients = [[]for i in range(n_shares)]
    for weights_layer in input_list:
        y_i = apply_poly(weights_layer.flatten(), n_shares, k)[1]
        for i in range(n_shares):
            list_clients[i].append(y_i[i])

    return list_clients


def apply_poly(S, N, K):
    """
    Function to perform secret sharing algorithm and encode the given secret when the secret is a tensor of values
    :param S: The secret to share (a tensor of values of type numpy.ndarray)
    :param N: the number of parts to divide the secret (so the number of participants)
    :param K: the minimum number of parts to reconstruct the secret, called threshold (so with a polynomial of order K-1)

    :return: the points generated from the polynomial we created for each part
    """
    # A tensor of polynomials to store the coefficients of each polynomial
    # The element i of the column 0 corresponds to the constant of the polynomial i which is the secret S_i
    # that we want to encrypt
    poly = np.array([[S[i]] + [0] * (K - 1) for i in range(len(S))])

    # Chose randomly K - 1 numbers for each row except the first column which is the secret
    poly[:, 1:] = np.random.randint(1, 996, np.shape(poly[:, 1:])) + 1

    # Generate N points for each polynomial we created
    points = np.array([
        [
            (x, calculate_y(x, poly[i])) for x in range(1, N + 1)
        ] for i in range(len(S))
    ]).T
    return points


# %% Functions to decrypt the secret
def sum_shares(encrypted_list, type_ss="additif"):
    """
    Function to sum the parts received by an entity
    :param encrypted_list: list of lists to decrypt
    :param type_ss: type of secret sharing algorithm used
    :param m: number of parts used to reconstruct the secret (M <= K)
    :return: the sum of the parts received
    """
    if type_ss == "additif":
        return sum_shares_additif(encrypted_list)

    elif type_ss == "shamir":
        return sum_shares_shamir(encrypted_list)

    else:
        raise ValueError("Type of secret sharing not recognized")


def sum_shares_additif(encrypted_list):
    """
    Function to sum the parts received by an entity when the secret sharing algorithm used is additif.
    :param encrypted_list: list of shares to sum
    :return: the sum of the parts received
    so decrypted_list[layer] = sum([weights_1[layer], weights_2[layer], ..., weights_n[layer]])
    """
    decrypted_list = []
    n_shares = len(encrypted_list)

    for layer in range(len(encrypted_list[0])):  # for each layer
        # We sum each part of the given layer
        sum_array = sum([encrypted_list[i][layer] for i in range(n_shares)])

        decrypted_list.append(sum_array)

    return decrypted_list


def sum_shares_shamir(encrypted_list):
    """
    Function to sum the parts received by an entity when the secret sharing algorithm used is Shamir.
    In this case, we sum points.
    :param encrypted_list: list of shares to sum where each element is a tuple (x, y) with x a abscissa and y all of the y received for this x.
    :return: the sum of the parts received so result_som[x] = sum([y_1, y_2, ..., y_n])
    """
    result_som = {}  # sum for a given client
    for (x, list_weights) in encrypted_list:
        if x in result_som:
            for layer in range(len(list_weights)):
                result_som[x][layer] += list_weights[layer]
        else:
            result_som[x] = list_weights

    return result_som


# %% Other functions
def data_preparation(filename, number_of_nodes=3):
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
    df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
    df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    df = df.dropna()

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    num_samples = len(df_shuffled)
    split_size = num_samples // number_of_nodes

    multi_df = []
    for i in range(number_of_nodes):
        if i < number_of_nodes - 1:
            subset = df.iloc[i * split_size:(i + 1) * split_size]
        else:
            subset = df.iloc[i * split_size:]

        x_s = subset.drop(['satisfaction'], axis=1)
        y_s = subset[['satisfaction']]

        multi_df.append([x_s, y_s])

    return multi_df


def save_nodes_chain(nodes):
    for node in nodes:
        node[0].blockchain.save_chain_in_file(node[0].id)


class Client:
    def __init__(self, id, host, port, batch_size, train, test, dp=False, type_ss="additif", threshold=3, m=3):
        self.id = id
        self.host = host
        self.port = port

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
                                                          stratify=y_train)

        self.flower_client = FlowerClient(batch_size, x_train, x_val, x_test, y_train, y_val, y_test,
                                          dp, delta=1 / (2 * len(x_train)))

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
        """
        Function to train the model
        :return:
        """
        old_params = self.flower_client.get_parameters({})
        res = old_params[:]
        for i in range(3):
            # why 3 iterations?
            res = self.flower_client.fit(res, {})[0]
            loss = self.flower_client.evaluate(res, {})[0]
            with open('output.txt', 'a') as f:
                f.write(f"client {self.id}: {loss} \n")

        # Apply SMPC (warning : list_shapes is initialized only after the first training)
        print(len(self.connections) + 1)
        encrypted_lists, self.list_shapes = apply_smpc(res, len(self.connections) + 1, self.type_ss, self.threshold)
        # we keep the last share of the secret for this client and send the others to the other clients
        self.frag_weights.append(encrypted_lists.pop())
        return encrypted_lists

    def send_frag_clients(self, frag_weights):
        """
        function to send the shares of the secret to the other clients
        :param frag_weights: list of shares of the secret (one share for each other client)
        """
        for i, (k, v) in enumerate(self.connections.items()):
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

    def send_frag_node(self):
        address = self.node.get('address')

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', address))

        serialized_data = pickle.dumps(self.sum_weights)  # The values summed on the client side and sent to the node.
        message = {"type": "frag_weights", "id": self.id, "value": serialized_data, "list_shapes": self.list_shapes}

        serialized_message = pickle.dumps(message)

        # Send the length of the message first
        client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
        client_socket.send(serialized_message)

        # Close the socket after sending
        client_socket.close()

        self.frag_weights = []

    def reset_connections(self):
        self.connections = {}

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
        return sum_shares(self.frag_weights, self.type_ss)

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)
