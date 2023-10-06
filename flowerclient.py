import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

from model import Model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = Model().to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

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
        for i in range(epoch):
            y_pred = self.model(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            loss_fn = torch.nn.CrossEntropyLoss()  # Utilisez la fonction de perte appropriée
            loss = loss_fn(y_pred, self.y_test)

            # Calcul de l'exactitude (accuracy)
            preds = torch.max(y_pred, dim=1)[1]
            correct = (preds == self.y_test).sum().item()
            total = self.y_test.shape[0]
            accuracy = correct / total * 100
        return loss.item(), accuracy
