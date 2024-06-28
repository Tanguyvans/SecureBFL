import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from going_modular.model import MobileNet
from collections import OrderedDict
import numpy as np
import os 

from flowerclient import FlowerClient
from going_modular.data_setup import load_dataset

device = torch.device("mps")

def splitting_cifar_dataset(dataset, nb_clients):
    labels_per_client = len(dataset) * 0.1 // nb_clients
    clients = [{i: 0 for i in range(10)} for _ in range(nb_clients)]
    clients_dataset = [[[], []] for _ in range(nb_clients)]

    for data in dataset:
        for i, client in enumerate(clients):
            if client[data[1]] < labels_per_client:
                client[data[1]] += 1
                clients_dataset[i][0].append(data[0])
                clients_dataset[i][1].append(data[1])
                break
    
    return clients_dataset

def load_data(partition_id, num_clients):
    dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True)
    dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True)

    train_sets = splitting_cifar_dataset(dataset_train, num_clients)
    test_sets = splitting_cifar_dataset(dataset_test, num_clients)

    x_train, y_train = train_sets[partition_id]
    train_data = TensorDataset(torch.stack([transforms.functional.to_tensor(img) for img in x_train]), torch.tensor(y_train))
    
    x_test, y_test = test_sets[partition_id]
    test_data = TensorDataset(torch.stack([transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))

    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, drop_last=True)

    return train_loader, test_loader

def get_model_files(directory): 
    all_files = os.listdir(directory)
    model_files = [file for file in all_files if file.endswith('.npz')]
    return model_files

def evaluate_model(model, dataloader, model_directory): 
    loaded_weights_dict = np.load(model_directory)
    loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

    params_dict = zip(model.state_dict().keys(), loaded_weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    test_acc = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(dataloader)
    test_acc = 100 * correct / total

    return {"loss": test_loss, "accuracy": test_acc}
    


if __name__ == "__main__":
    directory = "models/"

    data_root = "Data"
    name_dataset = "cifar"

    _, _, node_test_sets, n_classes = load_dataset(None, name_dataset, data_root, 1, 1)

    x_test, y_test = node_test_sets[0]      

    flower_client = FlowerClient.node(
            x_test=x_test, 
            y_test=y_test,
            batch_size=32,
            dp=False,
            name_dataset=name_dataset,
            model_choice="mobilenet",
            num_classes=10,
            choice_loss="cross_entropy",
            choice_optimizer="Adam",
            choice_scheduler=None
        )

    model_list = get_model_files(directory)

    print(model_list)

    model = MobileNet().to(device)

    for model_file in model_list: 
        loaded_weights_dict = np.load(directory+model_file)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        metrics = flower_client.evaluate(loaded_weights, {})

        print(metrics)

        # with open("evaluation.txt", "a") as file: 
        #     file.write(f"{model_file}: {metrics['loss']}, {metrics['accuracy']} \n")
