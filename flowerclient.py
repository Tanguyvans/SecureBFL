import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import Model
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size ,X_train, X_val, X_test, y_train, y_val ,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        train_data = Data(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))
        val_data = Data(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values))
        test_data = Data(torch.FloatTensor(X_test.values), torch.FloatTensor(y_test.values))

        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=1)

        self.model = Model().to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def get_dict_params(self, config): 
        return {f'param_{i}': val.cpu().numpy() for i, (_, val) in enumerate(self.model.state_dict().items())}

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.y_train), {'accuracy': accuracy}

    def train(self, epoch=1):
        self.model.train()
        for e in range(1, epoch+1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                
                y_pred = self.model(X_batch)
                
                loss = self.criterion(y_pred, y_batch)
                acc = binary_acc(y_pred, y_batch)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            self.model.eval()
            val_loss = 0
            val_acc = 0
            for X_val_batch, y_val_batch in self.val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = self.model(X_val_batch)

                val_batch_loss = self.criterion(y_val_pred, y_val_batch)
                val_batch_acc = binary_acc(y_val_pred, y_val_batch)

                val_loss += val_batch_loss.item()
                val_acc += val_batch_acc.item()

            epoch_loss /= len(self.train_loader)
            epoch_acc /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)

    def test(self):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        for X_test_batch, y_test_batch in self.test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)

            y_test_pred = self.model(X_test_batch)

            test_batch_loss = self.criterion(y_test_pred, y_test_batch)
            test_batch_acc = binary_acc(y_test_pred, y_test_batch)

            test_loss += test_batch_loss.item()
            test_acc += test_batch_acc.item()

        test_loss /= len(self.test_loader)
        test_acc /= len(self.test_loader)

        return test_loss, test_acc
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
