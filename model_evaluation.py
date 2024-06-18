import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from collections import OrderedDict
from model import CNNCifar  # Assuming you have a model class CNNCifar
import numpy as np

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
    test_loader = DataLoader(dataset=test_data, batch_size=256)

    return train_loader, test_loader

# Set device
device = torch.device("cpu")

trainloader, testloader = load_data(partition_id=0, num_clients=1)

# Initialize and load the model
model = CNNCifar().to(device)
model_directory = 'models/n1m137.npz'  # Update this path
loaded_weights_dict = np.load(model_directory)
loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

params_dict = zip(model.state_dict().keys(), loaded_weights)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
model.load_state_dict(state_dict, strict=True)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Evaluate the model
model.eval()
correct = 0
total = 0
total_loss = 0.0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

average_loss = total_loss / len(testloader)
accuracy = 100 * correct / total

print(f'Accuracy of the model on the 10000 test images: {accuracy}%')
print(f'Average loss on the test images: {average_loss}')