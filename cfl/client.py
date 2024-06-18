import argparse
import warnings
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CNNCifar

warnings.filterwarnings("ignore", category=UserWarning)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def train(model, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total = 0
        correct = 0
        for x_batch, y_batch in trainloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(x_batch)

            # Calculate loss
            loss = criterion(y_pred, y_batch)

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # Print the current loss and accuracy
            if total > 0:
                accuracy = 100 * correct / total
                #print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

def test(model, testloader):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(x_batch)

            # Calculate loss
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            # Calculate accuracy
            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def splitting_cifar_dataset(dataset, nb_clients):
    labels_per_client = len(dataset)*0.1//nb_clients
    clients = [{i:0 for i in range(10)} for _ in range(nb_clients)]
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
    model_choice = "CNNCifar"
    dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True)
    dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True)

    train_sets = splitting_cifar_dataset(dataset_train, num_clients)
    test_sets = splitting_cifar_dataset(dataset_test, num_clients)

    x_train, y_train = train_sets[partition_id]
    train_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_train]), torch.tensor(y_train))
    
    x_test, y_test = test_sets[partition_id]
    test_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))

    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=256)

    return train_loader, test_loader

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader, epochs=5)
        return self.get_parameters(config={}), len(trainloader.dataset), {}  # Use len(x_train) or len(y_train)


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}
    
    def save_model(self, model_path="global_model.pth"):
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", required=True, type=int, help="Partition of the dataset divided into 3 iid partitions created artificially.")
    parser.add_argument("--num-clients", required=True, type=int, help="Number of clients to simulate.")
    args = parser.parse_args()

    print(f"Number of clients: {args.num_clients}")
    
    model = CNNCifar().to(DEVICE)
    model = model.to(memory_format=torch.channels_last).to(DEVICE)
    trainloader, testloader = load_data(partition_id=args.partition_id, num_clients=args.num_clients)

    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient())

