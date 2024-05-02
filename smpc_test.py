# %%
from flowerclient import FlowerClient
from sklearn.model_selection import train_test_split
from client import (apply_smpc, data_preparation, decrypt_list_of_lists, decrypt_shamir, encrypt, decrypt_additif,
                    apply_shamir, sum_shares_shamir)
from node import aggregate_shamir, decrypt_shamir_node, combine_shares_node
import numpy as np
import random


def ss_shamir(list_secrets, k, n_shares=3):
    # On va simuler le partage de secret dans un ordre aléatoire entre les clients
    list_clients = [[] for i in range(n_shares)]
    # shamir : [num_client][layer][coordonnee]
    # additif secret sharing : [num_client][layer][line_tensor]
    secret_shape = [weights_layer.shape for weights_layer in list_secrets[0]]
    for i, secret_client in enumerate(list_secrets):
        encrypted_result = apply_shamir(secret_client, n_shares=n_shares, k=k)
        indices = [i for i in range(len(encrypted_result))]  # on fait une liste des indices pour les mélanger
        list_x = [i + 1 for i in range(len(encrypted_result))]  # on récupère les valeurs de x associées à chaque y
        random.shuffle(indices)  # on mélange les indices pour distribuer les parts de manière aléatoire
        for id_client in indices:
            list_clients[i].append((list_x[id_client], encrypted_result[id_client]))
    return list_clients, secret_shape


def aggregation_cluster(cluster_weight, secret_shape, m, clusters):
    # cluster_weights[id_share][1][layer]

    aggregated_weights = aggregate_shamir(cluster_weight, secret_shape, m)

    #loss = self.flower_client.evaluate(aggregated_weights, {})[0]

    cluster_weight = []
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
    type_ss = "shamir"  # "shamir" or "additif"
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
    # encrypted_result[num_client][layer][coordonnee]
    encrypted_result, dic_shapes = apply_smpc(res, n_shares, type_ss, k, m)
    # a = encrypted_result.pop()
    # %%
    decrypted_result = decrypt_list_of_lists(encrypted_result, type_ss)

    # %%
    print("Liste de listes originale :")
    print(res)

    # %%
    # print("\nListe de listes chiffrée :")
    # print(encrypted_result[0])
    # print(encrypted_result[1])
    # print(encrypted_result[2])

    print("\nListe de listes déchiffré :")
    print(decrypted_result)

    # %% verifier qu'on obtient bien le même résultat
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
    encrypted_list = [[] for i in range(n_shares)]
    for weights_layer in res:
        # verifier si c'est bien une matrice
        if isinstance(weights_layer[0], np.ndarray):
            # liste pour la couche donnée où chaque élement est une share pour un client.
            encrypted_layer = [[] for i in range(n_shares)]  # crypter un tenseur de poids
            for line in weights_layer:
                # liste pour une ligne d'un tenseur de poids d'une couche donnée où chaque élement est une share pour un client.
                encrypted_line = [[] for i in range(n_shares)]
                for elem in line:
                    # On crée un tuple où chaque élement est une share pour un client et leur somme est égale à elem
                    crypted_tuple = encrypt(elem, n_shares)
                    for i in range(n_shares):
                        # On a calculé les parts d'un element pour une ligne d'un tenseur de poids
                        # On ajoute chaque part d'un element à la ligne correspondante du client i
                        encrypted_line[i].append(crypted_tuple[i])

                # Pour une couche donnée, on attribue au client i la ligne cryptée
                for i in range(n_shares):
                    encrypted_layer[i].append(encrypted_line[i])
        else:
            encrypted_layer = [[] for i in range(n_shares)]
            for x in weights_layer:
                crypted_tuple = encrypt(x, n_shares)

                for i in range(n_shares):
                    encrypted_layer[i].append(crypted_tuple[i])

        # Au final, chaque client i a une part de chaque ligne de la couche donnée
        # Car on ajoute pour le client i, l'encrypted_layer qu'on a préparé pour ce client
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_layer[i]))

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # %% voir si on choisit bien les parts pour le secret sharing de Shamir
    a = decrypt_additif([encrypted_list[0], encrypted_list[1], encrypted_list[2]])
    b = decrypt_shamir([encrypted_result[0], encrypted_result[1]], dic_shapes)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # %% test de l'addition de polynomes et donc l'addition de shamir secret sharing
    S1 = res
    S2 = res
    S3 = res
    list_secrets = [S1, S2, S3]
    # %% tester sur un dictionnaire simple state_dict
    if m < k:
        print('Les points sont moins que le seuil de ' + str(k) + ', plus de points sont nécessaires')

    # %%Secret sharing
    list_clients, secret_shape = ss_shamir(list_secrets, k, n_shares)

    # %% on va faire pour un client donné la somme des parts qu'il a reçu pour chaque valeur de x
    secrets_result = []
    for id_client in range(len(list_clients)):
        result_som = sum_shares_shamir(list_clients[id_client])
        secrets_result.append(result_som)
    # %% on somme maintenant les sommes des 3 clients pour chaque valeur de x afin d'avoir un seul model à déchiffrer
    # On réunit les parts reçues au node.

    dico_secrets_combined = combine_shares_node(secrets_result)

    # %% on va decoder le secret final coté node
    # Secret Reconstruction
    decrypted_result = decrypt_shamir_node(dico_secrets_combined, secret_shape, m)

    # %% verifier qu'on obtient bien le même résultat
    for layer, val in enumerate(decrypted_result):
        print(f"Layer {layer} : {(val == decrypted_result[layer]).all()}")

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # %% tester direct avec les classes client et node
    list_frag_weights = [[] for i in range(n_shares)]
    #%%#############################  client 1 ################################
    for id_client, frag_weights in enumerate(list_frag_weights):
        print(f"Client {id_client} :")
        indices = [i for i in range(n_shares) if i != id_client]
        ############# train() function #############
        encrypted_lists, list_shapes = apply_smpc(res, n_shares, type_ss, k, m)
        # we keep the last share of the secret for this client and send the others to the other clients
        a = encrypted_lists.pop()
        print(f"in train, the id {id_client} keeps the share = {a[0]} for him.\n")
        list_frag_weights[id_client].append(a)

        ####### send_frag_clients() function #######
        print(
            f"in send_frag_clients, the id {id_client} sends the frag_weights = {[share[0] for share in encrypted_lists]}\n")

        for other_client in indices:
            ########### handle_message() function ###########
            #if message["type"] == "frag_weights":
            weights = encrypted_lists[0]
            print(f"client {other_client} received weights from {id_client} : {weights}")
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
            f"in send_frag_node for the id {id_client}, self.frag_weights = {[share[0] for share in list_frag_weights[id_client]]}\n"
            f"len(self.frag_weights)={len(list_frag_weights[id_client])} vaut logiquement 3 si tout va bien\n"
            f"len(self.frag_weights[0][0]) doit valoir 2 car tuple (x,y)  et vaut {len(list_frag_weights[id_client][0])}\n"
            f"a[abscisse][layer] donc type(a) doit être un dico de taille 3 si 3 abscisses diff reçus : len(a) = {len(a)}, type(a) = {type(a)}\n")

        print(f"serialized_data: {len(a)} == nbr_abcisse and {len(list(a.values())[0])} == nbr_layers\n")


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
                        f"logiquement, pour le cluster {pos}, weights correspond au result_som du client {id_client} donc weights[abcisse][layer],"
                        f"donc type(weights), len(weights[{test_abscisse}]) doivent être égaux à (dict, 4) : ({type(weights)}, {len(weights[test_abscisse])}\n)")
                    if len(weights[test_abscisse]) > 4:
                        print(
                            f"et taille des couches : {len(weights[test_abscisse][-1])} and len(weights[test_abscisse][-1][0] = {len(weights[test_abscisse][-1][0])}")
                    else:
                        print("ok ok")
                    cluster[id_client] = 1
                    cluster["count"] += 1

                if cluster["count"] == cluster["tot"]:
                    print(
                        "On est là",
                        len(cluster_weights), len(cluster_weights[pos]), len(cluster_weights[pos][0]), len(cluster_weights[pos][0][1])
                    )
                    aggregated_weights = aggregation_cluster(cluster_weights[pos], list_shapes, m, clusters[pos])

                    participants = [k for k in cluster.keys() if k not in ["count", "tot"]]
