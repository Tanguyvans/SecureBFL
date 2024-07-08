import logging
import warnings
import random
import numpy as np

from client import Client
from going_modular.data_setup import load_dataset

warnings.filterwarnings("ignore")

def train_client(client):
    frag_weights = client.train()

def encrypt(x, n_shares=2):
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)

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


# def apply_additif(input_list, n_shares=2):
#     """
#     Encrypt each tensor in the list of tensors.
#     """
#     encrypted_list = [[] for _ in range(n_shares)]
#     for tensor in input_list:
#         shares = encrypt(tensor, n_shares)
#         for i in range(n_shares):
#             encrypted_list[i].append(shares[i])
#     return encrypted_list

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

    # Validation step to check if the sum of shares equals the original input
    for layer_index, original_layer in enumerate(input_list):
        reconstructed_layer = sum([shares[layer_index] for shares in encrypted_list])

    return encrypted_list

def apply_no_smpc(input_list, n_shares): 
    return [input_list.copy() for _ in range(n_shares)]

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

def sum_shares_no_smpc(encrypted_list):
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
        sum_array = sum([encrypted_list[i][layer] for i in range(n_shares)])/n_shares

        decrypted_list.append(sum_array)

    return decrypted_list

# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    data_root = "Data"
    name_dataset = "cifar"  # Choose the dataset
    batch_size = 32
    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = "StepLR"
    learning_rate = 0.003
    epochs = 10 # Number of training epochs

    # Load dataset
    client_train_sets, client_test_sets, _, list_classes = load_dataset(None, name_dataset, data_root, 1, 1)

    # Create a single client
    client = Client(
        id="c1",
        host="127.0.0.1",
        port=5010,
        train=client_train_sets[0],
        test=client_test_sets[0],
        type_ss="additif",
        threshold=3,
        m=3,
        batch_size=batch_size,
        epochs=epochs,
        dp=False,  # Differential privacy setting
        name_dataset=name_dataset,
        model_choice="CNNCifar",  # Assuming CNNCifar is suitable for the cifar dataset
        classes=list_classes,
        choice_loss=choice_loss,
        learning_rate=learning_rate,
        choice_optimizer=choice_optimizer,
        choice_scheduler=choice_scheduler
    )

    # Train the client
    train_client(client)

    params = client.flower_client.get_parameters({})[:]
    print(client.flower_client.evaluate(params, {}))

    encrypted = apply_additif(params, 3)

    decrypted = sum_shares_additif(encrypted)
    
    are_close = all(np.allclose(p, d, atol=1e-5) for p, d in zip(params, decrypted))
    print("Parameters and decrypted values are approximately equal:", are_close)

    print(client.flower_client.evaluate(decrypted, {}))