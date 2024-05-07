# %%
from flowerclient import FlowerClient
from sklearn.model_selection import train_test_split
from client import (apply_smpc, data_preparation, sum_shares,
                    apply_shamir, sum_shares_shamir, apply_additif, sum_shares_additif)
from node import aggregate_shamir, decrypt_shamir_node, combine_shares_node
import numpy as np
import random


# %% Functions to implement the additif secret sharing
def encrypt_value(x, n_shares=3):
    """
    Function to encrypt a value with additif secret sharing
    :param x: the value to encrypt
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :return: the list of shares for each client
    """
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)


def apply_additif0(input_list, n_shares=3):
    """
    Function to apply SMPC to a list of lists

    :param input_list: List of lists to encrypt
    :param n_shares: Number of shares to create
    :return: The list of shares for each client
    """
    encrypted_list = [[] for i in range(n_shares)]
    for inner_list in input_list:
        if isinstance(inner_list[0], np.ndarray):
            encrypted_inner_list = [[] for i in range(n_shares)]
            for inner_inner_list in inner_list:
                encrypted_inner_inner_list = [[] for i in range(n_shares)]
                for x in inner_inner_list:
                    crypted_tuple = encrypt_value(x, n_shares)
                    for i in range(n_shares):
                        encrypted_inner_inner_list[i].append(crypted_tuple[i])

                for i in range(n_shares):
                    encrypted_inner_list[i].append(encrypted_inner_inner_list[i])
        else:
            encrypted_inner_list = [[] for i in range(n_shares)]
            for x in inner_list:
                crypted_tuple = encrypt_value(x, n_shares)

                for i in range(n_shares):
                    encrypted_inner_list[i].append(crypted_tuple[i])

        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_inner_list[i]))

    return encrypted_list


def sum_shares_additif0(encrypted_list):
    """
    Function to sum the shares of the secret for each client
    :param encrypted_list: list of lists to decrypt
    :return: the decrypted list
    """
    decrypted_list = []
    n_shares = len(encrypted_list)

    for i in range(len(encrypted_list[0])):
        sum_array = np.add(encrypted_list[0][i], encrypted_list[1][i])
        for j in range(2, n_shares):
            sum_array = np.add(sum_array, encrypted_list[j][i])

        decrypted_list.append(sum_array)

    return decrypted_list


# Function to implement the Shamir secret sharing
def ss_shamir(list_secrets, k=3, n_shares=3):
    """
    Function implementing Shamir's secret sharing by simulating the sharing of secrets between clients
    :param list_secrets: list of secrets for each client
    :param k: The threshold : The minimum number of parts to reconstruct the secret (so with a polynomial of order k-1)
    :param n_shares: The number of parts to divide the secret (so the number of participants)
    :return: the list of shares for each client
    """
    # We will simulate the sharing of the secret in a random order between the clients
    list_clients = [[] for i in range(n_shares)]
    # additif secret sharing : [num_client][layer][line_tensor]
    secret_shape = [weights_layer.shape for weights_layer in list_secrets[0]]
    for i, secret_client in enumerate(list_secrets):
        encrypted_result = apply_shamir(secret_client, n_shares=n_shares, k=k)
        indices = [i for i in range(len(encrypted_result))]  # We make a list of indices to shuffle them
        list_x = [i + 1 for i in range(len(encrypted_result))]  # We get the values of x associated with each y.
        random.shuffle(indices)  # We shuffle the indices to distribute the shares randomly.
        for id_client in indices:
            list_clients[i].append((list_x[id_client], encrypted_result[id_client]))
    return list_clients, secret_shape


# Function to aggregate the shares of the secret cluster side
def aggregation_cluster(cluster_weight, secret_shape, m, clusters):
    """
    Function to aggregate the shares of the secret cluster side? This is a copy of the function in the Node class.
    :param cluster_weight: All of the parts received by a cluster
    :param secret_shape: list of the shapes of the secret
    :param m: The number of parts used to reconstruct the secret (M <= K)
    :param clusters: the dictionary of the clusters
    :return: the aggregated weights
    """
    # cluster_weights[id_share][1][layer]

    aggregated_weights = aggregate_shamir(cluster_weight, secret_shape, m)

    for k, v in clusters.items():
        if k != "tot":
            clusters[k] = 0

    return aggregated_weights


# %%
if __name__ == '__main__':
    # %%
    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'
    n_shares = 3
    k = 3
    m = 3
    dp = False
    type_ss = "additif"  # "shamir" or "additif"
    # %%
    train_sets = data_preparation(train_path, 1)
    test_sets = data_preparation(test_path, 1)

    # %%
    x_train, y_train = train_sets[0]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                      stratify=y_train)
    x_test, y_test = test_sets[0]

    # %%
    flower_client = FlowerClient(256, x_train, x_val, x_test, y_train, y_val, y_test, dp, delta=1/(2 * len(x_train)))

    # %%
    old_params = flower_client.get_parameters({})
    res = old_params[:]
    for i in range(1):
        res = flower_client.fit(res, {})[0]
        loss = flower_client.evaluate(res, {})[0]

    # %% test with a simple dico easy to interpret
    state_dict = {
        "key1": np.array(
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]
            ]
        ),
        "key2": np.array(
            [
                [13, 14],
                [15, 16]
            ]
        ),
        "key3": np.array(
            [
                [17, 18],
                [19, 20],
            ]
        ),
        "key4": np.array(
            [
                [21],
            ]
        )
    }
    res = list(state_dict.values())
    # %%
    encrypted_result, dic_shapes = apply_smpc(res, n_shares, type_ss, k)

    # %%
    decrypted_result = sum_shares(encrypted_result, type_ss)

    # %% Check that we get the same result
    for layer, val in enumerate(res):
        print(f"Layer {layer} : {(np.round(np.float64(val),4) == decrypted_result[layer]).all()}")

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # %% test du additif secret sharing
    # encrypted_list[num_client][layer][line_tensor],
    encrypted_list1 = apply_additif0(res, n_shares)

    # %% Implement the additif secret sharing shorter
    encrypted_list2 = apply_additif(res, n_shares)

    # %% We recombine the shares to decrypt the secret
    decrypted_result1 = sum_shares_additif0(encrypted_list1)
    decrypted_result2 = sum_shares_additif(encrypted_list2)

    # %%
    precision = 2
    for layer, val in enumerate(decrypted_result2):
        print(
            f"Layer {layer} : {(np.round(np.float64(val), precision) == np.round(decrypted_result1[layer], precision)).all()}")
        if not (np.round(np.float64(val), precision) == np.round(decrypted_result1[layer], precision)).all():
            break

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # %% test de l'addition de polynômes et donc l'addition de shamir secret sharing
    # run these lines only if you have chosen the shamir secret sharing type
    S1 = res
    S2 = res
    S3 = res
    list_secrets = [S1, S2, S3]
    # %% Test on a simple dictionary state_dict
    if m < k:
        print('There are fewer points than the threshold of ' + str(k) + ' required, more points are needed')

    # %% Secret sharing
    list_clients, secret_shape = ss_shamir(list_secrets, k, n_shares)

    # %% For each client, we sum the shares he received for each value of x
    secrets_result = []
    for id_client in range(len(list_clients)):
        result_som = sum_shares_shamir(list_clients[id_client])
        secrets_result.append(result_som)
    # %% We sum the sums of the 3 clients for each value of x in order to have a single model to decrypt.
    # We combine the parts received at the node.
    dico_secrets_combined = combine_shares_node(secrets_result)

    # %% We will decode the final secret on the node side
    # Secret Reconstruction
    decrypted_result = decrypt_shamir_node(dico_secrets_combined, secret_shape, m)

    # %% Check that we get the same result
    for layer, val in enumerate(decrypted_result):
        print(f"Layer {layer} : {(val == decrypted_result[layer]).all()}")

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # %% Test directly with the client and node classes
    list_frag_weights = [[] for i in range(n_shares)]

    #%%#############################  client 1 ################################
    for id_client, frag_weights in enumerate(list_frag_weights):
        print(f"Client {id_client} :")
        indices = [i for i in range(n_shares) if i != id_client]
        ############# train() function #############
        encrypted_lists, list_shapes = apply_smpc(res, n_shares, type_ss, k)
        # we keep the last share of the secret for this client and send the others to the other clients
        a = encrypted_lists.pop()
        print(f"In train, the id {id_client} keeps the share = {a[0]} for him.\n")
        list_frag_weights[id_client].append(a)

        ####### send_frag_clients() function #######
        print(
            f"In send_frag_clients, the id {id_client} sends the frag_weights = {[share[0] for share in encrypted_lists]}\n")

        for other_client in indices:
            ########### handle_message() function ###########
            weights = encrypted_lists[0]
            print(f"Client {other_client} received weights from {id_client} : {weights}")
            list_frag_weights[other_client].append(weights)
            del encrypted_lists[0]


    #%%################################################  node 1 ###################################################
    clusters = [{client: 0 for client in [0, 1, 2]}]
    clusters[-1]["tot"] = len(clusters[-1])
    clusters[-1]["count"] = 0
    cluster_weights = [[] for i in range(n_shares)]
    for id_client in range(n_shares):
        print(f"Node 1  with client {id_client}:\n")
        indices = [i for i in range(n_shares) if i != id_client]
        ############# send_frag_node() function #############
        print("len(list_frag_weights[id_client]) = ", len(list_frag_weights[id_client]))
        a = sum_shares_shamir(list_frag_weights[id_client])
        print(
            f"In send_frag_node for the id {id_client}, self.frag_weights = {[share[0] for share in list_frag_weights[id_client]]}\n"
            f"len(self.frag_weights)={len(list_frag_weights[id_client])} should be equal to 3 if everything is fine\n"
            f"len(self.frag_weights[0][0]) should be equal to 2 because it's a tuple (x,y) and is equal to {len(list_frag_weights[id_client][0])}\n"
            f"a[abscisse][layer] so type(a) should be a dict of size 3 if 3 different abscissas received : len(a) = {len(a)}, type(a) = {type(a)}\n"
        )

        print(f"Serialized_data: {len(a)} == nbr_abcisse and {len(list(a.values())[0])} == nbr_layers\n")


        list_frag_weights[id_client] = []

        ########### handle_message() function ###########

        weights = a
        # print(f"node {self.id} received weights from {message_id} : {weights}")
        for pos, cluster in enumerate(clusters):

            if id_client in cluster:
                if cluster[id_client] == 0:
                    cluster_weights[pos].append(weights)
                    test_abscisse = list(weights.keys())[0]
                    print(
                        f"Logically, for the cluster {pos}, weights should correspond to the result_som of client {id_client} so weights[abscissa][layer],"
                        f"so type(weights), len(weights[{test_abscisse}]) should be equal to (dict, 4) : ({type(weights)}, {len(weights[test_abscisse])}\n)"
                    )

                    cluster[id_client] = 1
                    cluster["count"] += 1

                if cluster["count"] == cluster["tot"]:
                    print(
                        len(cluster_weights), len(cluster_weights[pos]), len(cluster_weights[pos][0]), len(cluster_weights[pos][0][1])
                    )
                    aggregated_weights = aggregation_cluster(cluster_weights[pos], list_shapes, m, clusters[pos])

                    participants = [k for k in cluster.keys() if k not in ["count", "tot"]]
