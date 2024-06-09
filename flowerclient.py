import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import Model, RnnNet, CNNCifar
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import DataLoader, TensorDataset
import torchvision

# For the differential privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator  # to validate our model with differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager

import numpy as np


def validate_dp_model(model_dp):
    """Check if the model is compatible with Opacus because it does not support all types of Pytorch layers"""
    if not ModuleValidator.validate(model_dp, strict=False):
        print("Model ok for the Differential privacy")
        return model_dp

    else:
        print("Model to be modified : ")
        return validate_dp_model(ModuleValidator.fix(model_dp))


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def sMAPE(outputs, targets):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) for evaluating the model.
    It is the sum of the absolute difference between the predicted and actual values divided by the average of
    the predicted and actual value, therefore giving a percentage measuring the amount of error :
    100/n * sum(|F_t - A_t| / (|F_t| + |A_t|) / 2) with t = 1 to n

    :param outputs : predicted values
    :param targets: real values
    :return: sMAPE
    """

    return 100 / len(targets) * np.sum(np.abs(outputs - targets) / (np.abs(outputs + targets)) / 2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, model_choice="simplenet", diff_privacy=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2, name_dataset="Airline Satisfaction"):
        self.batch_size = batch_size
        self.model_choice = model_choice
        self.diff_privacy = diff_privacy
        self.delta = delta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.name_dataset = name_dataset
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.len_train = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model after data loaders are potentially set
        self.model = self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    def initialize_model(self):
        if self.model_choice == "LSTM" or self.model_choice == "GRU":
            # Ensure data loaders are set before this point or handle it differently
            n_hidden = 25
            n_layers = 2
            input_dim = next(iter(self.train_loader))[0].shape[2] if self.train_loader else 10  # Default or error
            model = RnnNet(model_choice=self.model_choice, input_size=input_dim, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
            self.criterion = nn.MSELoss()
        elif self.model_choice == "CNNCifar":
            model = CNNCifar()
            self.criterion = nn.CrossEntropyLoss()
        else:
            model = Model()
            self.criterion = nn.BCEWithLogitsLoss()
        
        return validate_dp_model(model.to(self.device)) if self.diff_privacy else model.to(self.device)

    @classmethod
    def node(cls, batch_size, x_test, y_test, model_choice="simplenet", diff_privacy=True, epsilon=0.5, max_grad_norm=1.2, name_dataset="Airline Satisfaction"):
        obj = cls(batch_size, model_choice, diff_privacy, epsilon, max_grad_norm, name_dataset)
        # Set data loaders
        if name_dataset == "cifar":
            test_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))
        else:
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.test_loader = DataLoader(dataset=test_data, batch_size=1)
        return obj

    @classmethod
    def client(cls, batch_size, x_train, y_train, x_val, y_val, x_test, y_test, model_choice="simplenet", diff_privacy=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2, name_dataset="Airline Satisfaction"):
        obj = cls(batch_size, model_choice, diff_privacy, delta, epsilon, max_grad_norm, name_dataset)
        # Set data loaders
        if name_dataset == "cifar":
            train_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_train]), torch.tensor(y_train))
            val_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_val]), torch.tensor(y_val))
            test_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))
        else:
            train_data = TensorDataset(torch.FloatTensor(x_train).squeeze(-1), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(x_val).squeeze(-1), torch.FloatTensor(y_val))
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.len_train = len(y_train)
        obj.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        obj.val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        obj.test_loader = DataLoader(dataset=test_data, batch_size=1)
        return obj

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
        epochs = 1
        if self.diff_privacy:
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )

        epoch_loss, epoch_acc, val_loss, val_acc = self.train(epoch=epochs)

        return self.get_parameters(config={}), self.len_train, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test(self.test_loader)

        # list_outputs, list_targets, loss, accuracy = self.test(self.test_loader)

        # s_mape = round(sMAPE(np.array(list_outputs), np.array(list_targets)), 3)

        return float(loss), self.len_train, {'accuracy': accuracy}

    def train_step(self, dataloader):
        epoch_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:

            y_pred = self.model(x_batch, device=device)
            loss = self.criterion(y_pred, y_batch)

            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def train(self, epoch=1):
        for e in range(1, epoch+1):
            if self.diff_privacy:
                with BatchMemoryManager(data_loader=self.train_loader,
                                        max_physical_batch_size=int(self.batch_size / 4),
                                        optimizer=self.optimizer) as memory_safe_data_loader:
                    epoch_loss, epoch_acc = self.train_step(memory_safe_data_loader)

                    self.epsilon = self.privacy_engine.get_epsilon(self.delta)

            else:
                epoch_loss, epoch_acc = self.train_step(self.train_loader)

            val_loss, val_acc = self.test(self.val_loader)
            return epoch_loss, epoch_acc, val_loss, val_acc

    def test(self, dataloader, scaler=None, label_scaler=None):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        list_outputs = []
        list_targets = []
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images, device=device)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(dataloader)
        test_acc = 100 * correct / total
        return test_loss, test_acc


        # with torch.inference_mode():  # or with torch.no_grad():
        #     for x_test_batch, y_test_batch in dataloader:
        #         # Move data to device and set to float
        #         x_test_batch, y_test_batch = x_test_batch.float().to(device), y_test_batch.float().to(device)

        #         # Get model outputs
        #         y_test_pred = self.model(x_test_batch, device=device)
        #         if isinstance(self.model, RnnNet):
        #             # y_test_pred = y_test_pred.squeeze()
        #             # y_test_batch = y_test_batch.squeeze()
        #             if label_scaler:
        #                 y_test_pred = torch.tensor(scaler.inverse_transform(y_test_pred), device=device)
        #                 y_test_batch = torch.tensor(label_scaler.inverse_transform(y_test_batch), device=device)

        #         # print(y_test_batch, y_test_pred)
        #         list_targets.append(y_test_batch.detach().numpy())
        #         list_outputs.append(y_test_pred.detach().numpy())

        #         # Compute and accumulate metrics
        #         test_batch_loss = self.criterion(y_test_pred, y_test_batch)
        #         test_batch_acc = binary_acc(y_test_pred, y_test_batch)
        #         test_loss += test_batch_loss.item()
        #         test_acc += test_batch_acc.item()

        # test_loss /= len(dataloader)
        # test_acc /= len(dataloader)
        # self.model.train()

        return list_outputs, list_targets, test_loss, test_acc

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/len(y_test)  # correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
