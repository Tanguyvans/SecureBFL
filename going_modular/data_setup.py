from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
import torch
import os
import random
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'alzheimer': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    }


def data_preparation(data, name_dataset, number_of_nodes=3):
    multi_df = []
    if name_dataset == "Airline Satisfaction":
        df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

        num_samples = len(df_shuffled)
        split_size = num_samples // number_of_nodes
        for i in range(number_of_nodes):
            if i < number_of_nodes - 1:
                subset = data.iloc[i * split_size:(i + 1) * split_size]

            else:
                subset = data.iloc[i * split_size:]

            # x are the features and y the target
            x_s = subset.drop(['satisfaction'], axis=1)
            y_s = subset[['satisfaction']]

            multi_df.append([x_s, y_s])

    elif name_dataset == "Energy":
        num_samples = len(data[0])
        split_size = num_samples // number_of_nodes
        for i in range(number_of_nodes):
            if i < number_of_nodes - 1:
                # data : (x, y)
                x_s = data[0][i * split_size:(i + 1) * split_size]
                y_s = data[1][i * split_size:(i + 1) * split_size]

            else:
                x_s = data[0][i * split_size:]
                y_s = data[1][i * split_size:]

            multi_df.append([x_s, y_s])

    else:
        raise ValueError("Dataset not recognized")

    return multi_df


def create_sequences(data, seq_length):
    """
    Function to preprocess sequential data to make it usable for training neural networks.
    It transforms raw data into input-target pairs

    :param data: the dataframe containing the data or the numpy array containing the data
    :param seq_length: The length of the input sequences. It is the number of consecutive data points used as input to predict the next data point.
    :return: the numpy arrays of the inputs and the targets,
    where the inputs are sequences of consecutive data points and the targets are the immediate next data points.
    """
    if len(data) < seq_length:
        raise ValueError("The length of the data is less than the sequence length")

    xs, ys = [], []
    # Iterate over data indices
    for i in range(len(data) - seq_length):
        if type(data) is pd.DataFrame:
            # Define inputs
            x = data.iloc[i:i + seq_length]

            # Define target
            y = data.iloc[i + seq_length]

        else:
            # Define inputs
            x = data[i:i + seq_length]

            # Define target
            y = data[i + seq_length]

        xs.append(x)
        ys.append(y)

    # Convert lists to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)

    # Shuffle the sequences
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    xs = xs[indices]
    ys = ys[indices]

    return xs, ys


def get_dataset(filename, name_dataset="Airline Satisfaction"):
    if name_dataset == "Airline Satisfaction":
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0', 'id'], axis=1)
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
        df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        df = df.dropna()

    elif name_dataset == "Energy":
        df = pd.read_pickle(filename)
        df["ID"] = df["ID"].astype("category")
        df["time_code"] = df["time_code"].astype("uint16")
        df = df.set_index("date_time")

        # Electricity consumption per hour (date with hour in the index)
        df = df["consumption"].resample("60min", label='right', closed='right').sum().to_frame()

    else:
        raise ValueError("Dataset not recognized")

    return df


def splitting_dataset(dataset, nb_clients):
    """
    Split the CIFAR dataset into nb_clients clients
    :param dataset: torchvision dataset object
    :param nb_clients: number of clients to split the dataset into
    :return: list of clients datasets
    """
    num_classes = len(dataset.classes)
    labels_per_client = len(dataset) / num_classes // nb_clients

    clients = [{i: 0 for i in range(num_classes)} for _ in range(nb_clients)]
    clients_dataset = [[[], []] for _ in range(nb_clients)]

    for data in dataset:
        for i, client in enumerate(clients):
            if client[data[1]] < labels_per_client:
                client[data[1]] += 1
                clients_dataset[i][0].append(data[0])
                clients_dataset[i][1].append(data[1])
                break

    return clients_dataset


def split_data_client(dataset, num_clients, seed):
    """
        This function is used to split the dataset into train and test for each client.
        :param dataset: the dataset to split (type: torch.utils.data.Dataset)
        :param num_clients: the number of clients
        :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds


def load_dataset(resize=None, name_dataset="cifar", data_root="./data/", numberOfClientsPerNode=3, numberOfNodes=3):
    data_folder = f"{data_root}/{name_dataset}"
    classes = None
    if name_dataset == "Airline Satisfaction":
        train_path = f'{data_folder}/train.csv'
        test_path = f'{data_folder}/test.csv'
        dataset_train = get_dataset(train_path, name_dataset)
        dataset_test = get_dataset(test_path, name_dataset)

        client_train_sets = data_preparation(dataset_train, name_dataset, numberOfClientsPerNode * numberOfNodes)
        client_test_sets = data_preparation(dataset_test, name_dataset, numberOfClientsPerNode * numberOfNodes)

        node_test_sets = data_preparation(dataset_test, name_dataset, numberOfNodes)

    elif name_dataset == "Energy":
        client_train_sets = []
        client_test_sets = []
        node_test_sets = []

        IDs = set()
        for file in os.listdir(data_folder + "/Electricity"):
            if file[12] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                IDs.add(int(file[12:16]))
        ID_list = list(IDs)

        for client_id in range(numberOfClientsPerNode * numberOfNodes):
            if not ID_list:
                raise ValueError("Not enough IDs available for all clients.")
            selected_ID = random.choice(ID_list)  # Randomly select an ID
            ID_list.remove(selected_ID)  # Optional: remove to avoid reuse

            file_path = f"{data_folder}/Electricity/residential_{selected_ID}.pkl"
            df = get_dataset(file_path, name_dataset)
            window_size = 10
            dataset_train = create_sequences(df["2009-07-15": "2010-07-14"], window_size)
            dataset_test = create_sequences(df["2010-07-14": "2010-07-21"], window_size)

            client_train_sets.append(data_preparation(dataset_train, name_dataset, 1)[0])
            client_test_sets.append(data_preparation(dataset_test, name_dataset, 1)[0])

        for i in range(numberOfNodes):
            node_train_data = [[], []]
            node_test_data = [[], []]
            for j in range(5):

                if not ID_list:
                    raise ValueError("Not enough IDs available for all clients.")
                selected_ID = random.choice(ID_list)  # Randomly select an ID
                ID_list.remove(selected_ID)  # Optional: remove to avoid reuse

                file_path = f"{data_folder}/Electricity/residential_{selected_ID}.pkl"
                df = get_dataset(file_path, name_dataset)
                window_size = 10
                data_train = create_sequences(df["2009-07-15": "2010-07-14"], window_size)
                data_test = create_sequences(df["2010-07-14": "2010-07-21"], window_size)

                data_train_x, data_train_y = data_preparation(data_train, name_dataset, 1)[0]
                data_test_x, data_test_y = data_preparation(data_test, name_dataset, 1)[0]

                node_train_data[0].extend(data_train_x)
                node_train_data[1].extend(data_train_y)
                node_test_data[0].extend(data_test_x)
                node_test_data[1].extend(data_test_y)

            node_test_sets.append(node_test_data)

        del df, data_train, data_test

    else:
        # Case for the classification problems
        list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[name_dataset])]
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms
        transform = transforms.Compose(list_transforms)

        if name_dataset == "cifar":
            dataset_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
            dataset_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)

        elif name_dataset == "mnist":
            dataset_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
            dataset_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

        elif name_dataset == "alzheimer":
            dataset_train = datasets.ImageFolder(data_folder + "/train", transform=transform)
            dataset_test = datasets.ImageFolder(data_folder + "/test", transform=transform)

        else:
            raise ValueError("The dataset name is not correct")

        client_train_sets = splitting_dataset(dataset_train, numberOfClientsPerNode * numberOfNodes)
        client_test_sets = splitting_dataset(dataset_test, numberOfClientsPerNode * numberOfNodes)
        # client_train_sets = split_data_client(dataset_train, numberOfClientsPerNode * numberOfNodes, seed=42)
        # client_test_sets = split_data_client(dataset_test, numberOfClientsPerNode * numberOfNodes, seed=42)

        node_test_sets = splitting_dataset(dataset_test, numberOfNodes)
        # node_test_sets = split_data_client(dataset_test, numberOfNodes, seed=42)

        classes = dataset_train.classes

    return client_train_sets, client_test_sets, node_test_sets, classes


def load_data(partition_id, num_clients, name_dataset="cifar", data_root="./data", resize=None, batch_size=256):
    data_folder = f"{data_root}/{name_dataset}"
    # Case for the classification problems
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[name_dataset])]
    if resize is not None:
        list_transforms = [transforms.Resize((resize, resize))] + list_transforms
    transform = transforms.Compose(list_transforms)

    if name_dataset == "cifar":
        dataset_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)

    elif name_dataset == "mnist":
        dataset_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

    elif name_dataset == "alzheimer":
        dataset_train = datasets.ImageFolder(data_folder + "/train", transform=transform)
        dataset_test = datasets.ImageFolder(data_folder + "/test", transform=transform)

    else:
        raise ValueError("The dataset name is not correct")

    train_sets = splitting_dataset(dataset_train, num_clients)
    test_sets = splitting_dataset(dataset_test, num_clients)

    x_train, y_train = train_sets[partition_id]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=None)
    train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
    val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))

    x_test, y_test = test_sets[partition_id]
    test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_data, batch_size=batch_size)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size)

    return trainloader, valloader, testloader, dataset_train.classes


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)