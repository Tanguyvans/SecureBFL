from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import random

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'alzheimer': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'caltech256': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


def splitting_dataset(dataset, nb_clients):
    random.seed(42)
    
    # Group data by class
    class_data = {}
    for data, target in dataset:
        if target not in class_data:
            class_data[target] = []
        class_data[target].append((data, target))
    
    # Initialize client datasets
    clients_dataset = [[[], []] for _ in range(nb_clients)]
    
    # First pass: distribute samples from each class evenly
    for class_label, samples in class_data.items():
        samples = samples.copy()
        random.shuffle(samples)
        
        # Ensure each client gets at least one sample from each class if possible
        if len(samples) >= nb_clients:
            for client_idx in range(nb_clients):
                data, target = samples.pop()
                clients_dataset[client_idx][0].append(data)
                clients_dataset[client_idx][1].append(target)
        
        # Distribute remaining samples round-robin
        while samples:
            for client_idx in range(nb_clients):
                if not samples:
                    break
                data, target = samples.pop()
                clients_dataset[client_idx][0].append(data)
                clients_dataset[client_idx][1].append(target)
    
    # Verify and print distribution
    print("\nOverall dataset distribution:")
    total_distributed = sum(len(x) for x, _ in clients_dataset)
    print(f"Total samples: {total_distributed}")
    print(f"Average samples per client: {total_distributed/nb_clients:.2f}")
    print(f"Number of clients: {nb_clients}")
    
    # Print per-client distribution
    print("\nPer-client distribution:")
    empty_clients = []
    for i, (x, y) in enumerate(clients_dataset):
        if len(x) == 0:
            empty_clients.append(i)
            continue
            
        class_dist = {}
        for label in y:
            class_dist[label] = class_dist.get(label, 0) + 1
        print(f"Client {i}:")
        print(f"  Total samples: {len(x)}")
        print(f"  Number of classes: {len(class_dist)}")
        print(f"  Samples per class: {dict(sorted(class_dist.items()))}")
    
    # Handle empty clients by redistributing data
    if empty_clients:
        print(f"\nWarning: Found {len(empty_clients)} empty clients. Redistributing data...")
        for empty_client in empty_clients:
            # Find client with most samples
            max_client = max(range(nb_clients), 
                           key=lambda x: len(clients_dataset[x][0]) if x not in empty_clients else -1)
            
            # Transfer half of the samples
            n_transfer = len(clients_dataset[max_client][0]) // 2
            clients_dataset[empty_client][0] = clients_dataset[max_client][0][:n_transfer]
            clients_dataset[empty_client][1] = clients_dataset[max_client][1][:n_transfer]
            clients_dataset[max_client][0] = clients_dataset[max_client][0][n_transfer:]
            clients_dataset[max_client][1] = clients_dataset[max_client][1][n_transfer:]
    
    # Final verification
    client_sizes = [len(x) for x, _ in clients_dataset]
    size_difference = max(client_sizes) - min(client_sizes)
    if size_difference > nb_clients:
        print(f"\nWarning: Uneven distribution detected. Size difference: {size_difference}")
        print(f"Client sizes: {client_sizes}")
    
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


def load_data_from_path(resize=None, name_dataset="cifar10", data_root="./data/"):
    data_folder = f"{data_root}/{name_dataset}"
    
    if name_dataset == "caltech256":
        # For Caltech256, enforce resize if not specified
        if resize is None:
            resize = 224  # Standard size used in many models
            
        list_transforms = [
            transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB
            transforms.Resize((resize, resize)),  # Force resize
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[name_dataset])
        ]
    else:
        list_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[name_dataset])
        ]
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms
            
    transform = transforms.Compose(list_transforms)

    if name_dataset == "cifar10":
        dataset_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)
    elif name_dataset == "cifar100":
        dataset_train = datasets.CIFAR100(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR100(data_folder, train=False, download=True, transform=transform)
    elif name_dataset == "mnist":
        dataset_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)
    elif name_dataset == "alzheimer":
        dataset_train = datasets.ImageFolder(data_folder + "/train", transform=transform)
        dataset_test = datasets.ImageFolder(data_folder + "/test", transform=transform)
    elif name_dataset == "caltech256":
        full_dataset = datasets.Caltech256(data_folder, download=True, transform=transform)
        
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        dataset_train, dataset_test = random_split(full_dataset, [train_size, test_size],
                                                 generator=torch.Generator().manual_seed(42))
        # Store classes information using categories instead of classes
        dataset_train.classes = full_dataset.categories
        dataset_test.classes = full_dataset.categories

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
