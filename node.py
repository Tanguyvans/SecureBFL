import grpc
import block_pb2 as pb2
import block_pb2_grpc as pb2_grpc
import concurrent.futures
import hashlib
import numpy as np
from block import Block
from blockchain import Blockchain
from collections import OrderedDict

import flwr as fl

from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = Net().to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def get_dict_params(self, config): 
        return {f'param_{i}': val.cpu().numpy() for i, (_, val) in enumerate(self.net.state_dict().items())}

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {'accuracy': accuracy}

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for data, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for data, labels in testloader:
            outputs = net(data)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def load_data(): 
    custom_data = torch.randn(32, 5)
    classification_labels = torch.randint(0, 2, (32,))
    dataset = TensorDataset(custom_data, classification_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader, dataloader

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 2)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return x

class NodeServer(pb2_grpc.NodeServiceServicer):
    def __init__(self, port, id):
        self.id = id
        self.port = port
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'127.0.0.1:{self.port}')
        self.server.start()

        self.blockchain = Blockchain()
        self.trainloader, self.testloader = load_data()
        self.flower_client = FlowerClient(self.trainloader, self.testloader)
        self.params_directories = []

    def MineBlock(self): 
        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset

        filename = f"models/m{self.blockchain.len_chain}.npz"

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        loaded_weights_dict = np.load(filename)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])

        hash_model = hashlib.sha256()
        hash_model.update(str(loaded_weights).encode('utf-8'))
        hash_model = hash_model.hexdigest()

        block = Block(calculated_hash=hash_model, storage_reference=filename, previous_block=self.blockchain.blocks[-1])
        self.blockchain.add_block(block, block.cryptographic_hash)
        self.params_directories.append(block.storage_reference)

        return block
    
    def AddBlock(self, request, context): 
        new_block = Block(request.calculated_hash, request.storage_reference, self.blockchain.blocks[-1])
        self.blockchain.add_block(new_block, request.hash)
        self.params_directories.append(new_block.storage_reference)
        return pb2.Response(success=True, message=f"Block {request.block_number} added to the blockchain successfully {request.hash}")

    def aggregateParameters(self):
        params_list = []

        for cnt, model_directory in enumerate(self.params_directories):
            loaded_weights_dict = np.load(model_directory)
            loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
            print(f"model{cnt}: ", self.flower_client.evaluate(loaded_weights, {}))
            loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])
            params_list.append(loaded_weights)

        print("final evaluation: ",self.flower_client.evaluate(params_list[0][0], {}))
        self.aggregated_params = aggregate(params_list)


class NodeClient:
    def __init__(self):
        self.connections = {}

    def clientConnection(self, server): 
        channel = grpc.insecure_channel(f'127.0.0.1:{server.port}')  # Connectez-vous au nœud 2
        stub = pb2_grpc.NodeServiceStub(channel)
        self.connections[server.id] = stub 
  
    def broadcast_block(self, block):
        block_message = pb2.BlockMessage()
        block_message.nonce = block.nonce
        block_message.previous_hash = block.previous_block_cryptographic_hash
        block_message.hash = block.cryptographic_hash
        block_message.calculated_hash = block.calculated_hash
        block_message.storage_reference = block.storage_reference
        block_message.block_number = block.block_number
        
        responses = {}
        for k, v in self.connections.items(): 
            response = v.AddBlock(block_message)

            if response.success:
                print(f"Block added to {k}'s blockchain: {response.message}")
            else:
                print(f"Failed to add block to {k}'s blockchain: {response.message}")

            responses[k] = response

        return responses

if __name__ == "__main__":
    NodeServer1 = NodeServer("1234", "node1")
    NodeClient1 = NodeClient()
    NodeServer2 = NodeServer("2345", "node2")
    NodeClient2 = NodeClient()

    NodeClient1.clientConnection(NodeServer2)
    NodeClient2.clientConnection(NodeServer1)
