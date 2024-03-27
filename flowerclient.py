import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import Model
from torch.utils.data import Dataset, DataLoader

# For the differential privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator  # to validate our model with differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, x_train, x_val, x_test, y_train, y_val, y_test,
                 diff_privacy=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        train_data = Data(torch.FloatTensor(x_train.values), torch.FloatTensor(y_train.values))
        val_data = Data(torch.FloatTensor(x_val.values), torch.FloatTensor(y_val.values))
        test_data = Data(torch.FloatTensor(x_test.values), torch.FloatTensor(y_test.values))

        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=1)
        self.batch_size = batch_size

        self.model = validate_dp_model(Model().to(device)) if diff_privacy else Model().to(device)
        self.criterion = nn.BCEWithLogitsLoss()
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
        loss, accuracy = self.test(self.test_loader)
        return float(loss), len(self.y_train), {'accuracy': accuracy}

    def train_step(self, dataloader):
        epoch_loss = 0
        epoch_acc = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            self.optimizer.zero_grad()

            y_pred = self.model(x_batch)

            loss = self.criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

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

            val_loss, val_acc = self.test(self.val_loader)
            return epoch_loss, epoch_acc, val_loss, val_acc

    def test(self, dataloader):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for x_test_batch, y_test_batch in dataloader:
                x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)

                y_test_pred = self.model(x_test_batch)

                test_batch_loss = self.criterion(y_test_pred, y_test_batch)
                test_batch_acc = binary_acc(y_test_pred, y_test_batch)

                test_loss += test_batch_loss.item()
                test_acc += test_batch_acc.item()

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        self.model.train()
        return test_loss, test_acc


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
