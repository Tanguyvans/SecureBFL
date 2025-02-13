import numpy as np

# def validate_dp_model(model_dp):
#     """Check if the model is compatible with Opacus because it does not support all types of Pytorch layers"""
#     if not ModuleValidator.validate(model_dp, strict=False):
#         print("Model ok for the Differential privacy")
#         return model_dp

#     else:
#         print("Model to be modified : ")
#         return validate_dp_model(ModuleValidator.fix(model_dp))

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

        # For each layer, each client has a part of each row of the given layer
        # because we add for client i, the encrypted_layer we prepared for this client
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_layer[i]))

    return encrypted_list

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
    :param k: the minimum number of parts to reconstruct the secret, called threshold (with a polynomial of order k-1)
    :return: the list of shares for each client
    """
    list_clients = [[]for _ in range(n_shares)]
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
    :param K: the minimum number of parts to reconstruct the secret, called threshold (with a polynomial of order K-1)

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
    :param encrypted_list: list of shares to sum where each element is a tuple (x, y) with x an abscissa and all y received for this x.
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


def sum_shares(encrypted_list, type_ss="additif"):
    """
    Function to sum the parts received by an entity
    :param encrypted_list: list of lists to decrypt
    :param type_ss: type of secret sharing algorithm used
    :return: the sum of the parts received
    """
    if type_ss == "additif":
        return sum_shares_additif(encrypted_list)

    elif type_ss == "shamir":
        return sum_shares_shamir(encrypted_list)

    else:
        raise ValueError("Type of secret sharing not recognized")

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

def data_poisoning(data, poisoning_type, n_classes, poisoned_number, number_of_nodes=1, clients_per_node=1, target_class=None):

    if poisoning_type == "rand":
        poison_per_node = poisoned_number // number_of_nodes
        for node in range(number_of_nodes):
            for i in range(poison_per_node):
                client_index = node * clients_per_node + i
                data[client_index][1] = random_poisoning(n_classes, n=len(data[client_index][1]))
    
    elif poisoning_type == "targeted":
        # Targeted poisoning: poison only the first class
        for i in range(poisoned_number):
            data[i][1] = targeted_poisoning(data[i][1], n_classes)
    
    else:
        raise ValueError(f"Unknown poisoning type: {poisoning_type}. Choose from: 'rand' or 'targeted'")
    

def random_poisoning(n_classes, n):
    return np.random.randint(0, n_classes, size=n).tolist()

def targeted_poisoning(labels, n_classes, target_class=0):
    poisoned_labels = []
    for label in labels:
        if label == target_class:
            # Generate a random wrong label for the target class
            wrong_labels = list(range(n_classes))
            wrong_labels.remove(target_class)  # Remove the correct label
            poisoned_labels.append(np.random.choice(wrong_labels))
        else:
            poisoned_labels.append(label)
    return poisoned_labels