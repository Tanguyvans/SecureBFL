import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import Model, RnnNet
from torch.utils.data import Dataset, DataLoader

# For the differential privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator  # to validate our model with differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager

from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError
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
    def __init__(self, batch_size, x_train, x_val, x_test, y_train, y_val, y_test,
                 diff_privacy=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2,
                 name_dataset="Airline Satisfaction", model_choice="simplenet"):
        self.x_train = x_train if name_dataset == "Energy" else x_train.values
        self.x_test = x_test if name_dataset == "Energy" else x_test.values
        self.y_train = y_train if name_dataset == "Energy" else y_train.values
        self.y_test = y_test if name_dataset == "Energy" else y_test.values
        x_val = x_val if name_dataset == "Energy" else x_val.values
        y_val = y_val if name_dataset == "Energy" else y_val.values

        train_data = Data(torch.FloatTensor(self.x_train), torch.FloatTensor(self.y_train))
        val_data = Data(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
        test_data = Data(torch.FloatTensor(self.x_test), torch.FloatTensor(self.y_test))

        bool_shuffle = False if name_dataset == "Energy" else True
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=bool_shuffle)
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=bool_shuffle)
        self.test_loader = DataLoader(dataset=test_data, batch_size=1)
        self.batch_size = batch_size
        if model_choice == "LSTM" or model_choice == "GRU":
            n_hidden = 25  # number of features in the hidden state h  # 32
            n_layers = 2  # number of recurrent layers.
            input_dim = next(iter(self.train_loader))[0].shape[2]  # 5, so number of features in the input x
            self.criterion = nn.MSELoss()
            model = RnnNet(model_choice=model_choice, input_size=input_dim, hidden_size=n_hidden, num_layers=n_layers,
                           batch_first=True)

        else:
            model = Model()
            self.criterion = nn.BCEWithLogitsLoss()

        self.model = validate_dp_model(model.to(device)) if diff_privacy else model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.diff_privacy = diff_privacy
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

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

        return self.get_parameters(config={}), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        list_outputs, list_targets, loss, accuracy = self.test(self.test_loader)

        s_mape = round(sMAPE(np.array(list_outputs), np.array(list_targets)), 3)

        return float(loss), len(self.y_train), {'accuracy': accuracy}

    def train_step(self, dataloader):
        epoch_loss = 0
        epoch_acc = 0
        for x_batch, y_batch in dataloader:
            # Move data to device and set to float
            x_batch, y_batch = x_batch.float().to(device), y_batch.float().to(device)

            # Get model outputs
            y_pred = self.model(x_batch, device=device)

            # Compute and accumulate metrics
            loss = self.criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

        # Adjust metrics to get average loss and accuracy per batch
        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
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

            _, _, val_loss, val_acc = self.test(self.val_loader)
            return epoch_loss, epoch_acc, val_loss, val_acc

    def test(self, dataloader, scaler=None, label_scaler=None):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        list_outputs = []
        list_targets = []
        with torch.inference_mode():  # or with torch.no_grad():
            for x_test_batch, y_test_batch in dataloader:
                # Move data to device and set to float
                x_test_batch, y_test_batch = x_test_batch.float().to(device), y_test_batch.float().to(device)

                # Get model outputs
                y_test_pred = self.model(x_test_batch, device=device)
                if isinstance(self.model, RnnNet):
                    # y_test_pred = y_test_pred.squeeze()
                    # y_test_batch = y_test_batch.squeeze()
                    if label_scaler:
                        y_test_pred = torch.tensor(scaler.inverse_transform(y_test_pred), device=device)
                        y_test_batch = torch.tensor(label_scaler.inverse_transform(y_test_batch), device=device)

                # print(y_test_batch, y_test_pred)
                list_targets.append(y_test_batch.detach().numpy())
                list_outputs.append(y_test_pred.detach().numpy())

                # Compute and accumulate metrics
                test_batch_loss = self.criterion(y_test_pred, y_test_batch)
                test_batch_acc = binary_acc(y_test_pred, y_test_batch)
                test_loss += test_batch_loss.item()
                test_acc += test_batch_acc.item()

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        self.model.train()
        #return np.array(list_outputs).flatten(), np.array(list_targets).flatten(), test_loss, test_acc
        return list_outputs, list_targets, test_loss, test_acc


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/len(y_test)  # correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
