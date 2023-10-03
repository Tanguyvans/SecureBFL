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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from sklearn.datasets import make_classification

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data(Dataset):
  def __init__(self, X_train, y_train):
    self.X = torch.from_numpy(X_train.astype(np.float32))
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, network_structure):
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = Net(network_structure).to(DEVICE)

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

class Net(nn.Module):
  def __init__(self, network_structure):
    super(Net, self).__init__()
    self.linear1 = nn.Linear(network_structure["input_dim"], network_structure["hidden_layers"])
    self.linear2 = nn.Linear(network_structure["hidden_layers"], network_structure["output_dim"])
  def forward(self, x):
    x = torch.sigmoid(self.linear1(x))
    x = self.linear2(x)
    return x

class NodeServer(pb2_grpc.NodeServiceServicer):
    def __init__(self, port, id):
        self.id = id
        self.port = port
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'127.0.0.1:{self.port}')
        self.server.start()

        X, Y = make_classification(n_samples=100, n_features=4, n_redundant=0, n_informative=3,  n_clusters_per_class=2, n_classes=3)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        traindata = Data(X_train, Y_train)
        trainloader = DataLoader(traindata, shuffle=True, num_workers=2)

        testdata = Data(X_test, Y_test)
        testloader = DataLoader(testdata, shuffle=True, num_workers=2)

        network_structure = {
            "input_dim": 4,
            "hidden_layers": 25,
            "output_dim": 3
        }

        self.blockchain = Blockchain()
        self.flower_client = FlowerClient(trainloader, testloader, network_structure)
        self.params_directories = []

    def MineBlock(self, filename, model_type): 
        loaded_weights_dict = np.load(filename)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])

        hash_model = hashlib.sha256()
        hash_model.update(str(loaded_weights).encode('utf-8'))
        hash_model = hash_model.hexdigest()

        block = Block(calculated_hash=hash_model, storage_reference=filename, model_type=model_type, previous_block=self.blockchain.blocks[-1])

        return block
    
    def AppendBlock(self, block): 
        self.blockchain.add_block(block, block.cryptographic_hash)
        if block.model_type == "global_model": 
            self.global_params_directory = block.storage_reference
        else:
            self.params_directories.append(block.storage_reference)

    def CreateGlobalModel(self): 
        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        model_type = "global_model"

        filename = f"models/m{self.blockchain.len_chain}.npz"
        self.global_params_directory = filename

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        return block
    
    def UpdateModel(self):
        loaded_weights_dict = np.load(self.global_params_directory)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        self.flower_client.set_parameters(loaded_weights)

        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        model_type = "update"

        filename = f"models/m{self.blockchain.len_chain}.npz"

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        return block

    def AddBlockRequest(self, request, context): 
        new_block = Block(request.calculated_hash, request.storage_reference, request.model_type, self.blockchain.blocks[-1])
        if self.blockchain.is_valid_block(new_block, request.hash): 
            self.block = new_block
            if request.model_type == "global_model":
                return pb2.Response(success=True, message=f"Block {request.block_number} added to the blockchain successfully {request.hash}")
            else: 
                if self.isModelUpdateUsefull(new_block.storage_reference):
                    return pb2.Response(success=True, message=f"Block {request.block_number} added to the blockchain successfully {request.hash}")
                else: 
                    return pb2.Response(success=False, message=f"Block {request.block_number} was not added, loss too high")
        else: 
            return pb2.Response(success=False, message=f"Block {request.block_number} was not added for wrong hash")
        
    def AddBlockToChain(self, request, constext):
        self.blockchain.add_block(self.block, self.block.cryptographic_hash)
        if self.block.model_type == "global_model": 
            self.global_params_directory = self.block.storage_reference
        else:
            self.params_directories.append(self.block.storage_reference)

        return pb2.Response(success=True)
    
    def aggregateParameters(self):
        params_list = []

        for cnt, model_directory in enumerate(self.params_directories):
            loaded_weights_dict = np.load(model_directory)
            loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

            loss = self.flower_client.evaluate(loaded_weights, {})[0]
            acc = self.flower_client.evaluate(loaded_weights, {})[2]['accuracy']

            print(f"model{cnt}: loss: {loss}, acc: {acc}")
            with open("output.txt", "a") as f: 
                f.write(f"model{cnt}: loss: {loss}, acc: {acc} \n")
            loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])
            params_list.append(loaded_weights)

        self.aggregated_params = aggregate(params_list)
        self.flower_client.set_parameters(self.aggregated_params)

        final_loss = self.flower_client.evaluate(self.aggregated_params, {})[0]
        final_acc = self.flower_client.evaluate(self.aggregated_params, {})[2]['accuracy']
        print(f"final_model: loss: {final_loss}, acc: {final_acc}")
        with open("output.txt", "a") as f: 
            f.write(f"final_model: loss: {final_loss}, acc: {final_acc} \n")
        

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 40
        model_type = "global_model"

        filename = f"models/m{self.blockchain.len_chain}.npz"
        self.global_params_directory = filename

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        self.params_directories = []

        return block

    def isModelUpdateUsefull(self, model_directory): 
        if self.evaluateModel(model_directory) <= self.evaluateModel(self.global_params_directory)*0.95: 
            return True
        else: 
            print("no usefull")
            return False

    def evaluateModel(self, model_directory):
        loaded_weights_dict = np.load(model_directory)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loss = self.flower_client.evaluate(loaded_weights, {})[0]

        return loss

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
        block_message.model_type = block.model_type
        block_message.calculated_hash = block.calculated_hash
        block_message.storage_reference = block.storage_reference
        block_message.block_number = block.block_number
        
        responses = {'success': 0, 'failure': 0, 'tot': 0}
        for k, v in self.connections.items(): 
            response = v.AddBlockRequest(block_message)

            if response.success:
                responses['success'] += 1
            else:
                responses['success'] += 1
            responses['tot'] += 1

        return responses

    def broadcast_decision(self):
        block_message = pb2.BlockValidation()
        block_message.valid = True

        responses = {'success': 0, 'failure': 0, 'tot': 0}
        for k, v in self.connections.items(): 
            response = v.AddBlockToChain(block_message)

            if response.success:
                responses['success'] += 1
            else:
                responses['success'] += 1
            responses['tot'] += 1

        return responses
    
if __name__ == "__main__":
    NodeServer1 = NodeServer("1234", "node1")
    NodeClient1 = NodeClient()
    NodeServer2 = NodeServer("2345", "node2")
    NodeClient2 = NodeClient()

    NodeClient1.clientConnection(NodeServer2)
    NodeClient2.clientConnection(NodeServer1)
