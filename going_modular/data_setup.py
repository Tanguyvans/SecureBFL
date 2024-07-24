from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'alzheimer': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    }


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


def load_data_from_path(resize=None, name_dataset="cifar", data_root="./data/"):
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

    return dataset_train, dataset_test


def load_dataset(resize=None, name_dataset="cifar", data_root="./data/", number_of_clients_per_node=3, number_of_nodes=3):
    # Case for the classification problems
    dataset_train, dataset_test = load_data_from_path(resize, name_dataset, data_root)

    client_train_sets = splitting_dataset(dataset_train, number_of_clients_per_node * number_of_nodes)
    client_test_sets = splitting_dataset(dataset_test, number_of_clients_per_node * number_of_nodes)
    # client_train_sets = split_data_client(dataset_train, number_of_clients_per_node * number_of_nodes, seed=42)
    # client_test_sets = split_data_client(dataset_test, number_of_clients_per_node * number_of_nodes, seed=42)

    node_test_sets = splitting_dataset(dataset_test, number_of_nodes)
    # node_test_sets = split_data_client(dataset_test, number_of_nodes, seed=42)

    classes = dataset_train.classes

    return client_train_sets, client_test_sets, node_test_sets, classes


def load_data(partition_id, num_clients, name_dataset="cifar", data_root="./data", resize=None, batch_size=256):
    dataset_train, dataset_test = load_data_from_path(resize, name_dataset, data_root)

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
