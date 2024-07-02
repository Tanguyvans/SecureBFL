from collections import OrderedDict
import numpy as np
import torch
import flwr as fl

from torch.utils.data._utils.collate import default_collate

from going_modular.model import Model, RnnNet, Net

from going_modular.security import PrivacyEngine, validate_dp_model
from going_modular.data_setup import TensorDataset, DataLoader
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct, save_graphs,
                                 save_matrix, save_roc)
from going_modular.engine import train, test

import os


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, epochs=1, model_choice="simplenet", dp=True, delta=1e-5, epsilon=0.5,
                 max_grad_norm=1.2, name_dataset="Airline Satisfaction", device="gpu", classes=None,
                 learning_rate=0.001, choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None,
                 save_results=None, matrix_path=None, roc_path=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_choice = model_choice
        self.dp = dp
        self.delta = delta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.name_dataset = name_dataset
        self.learning_rate = learning_rate
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.len_train = None
        self.device = choice_device(device)
        self.classes = classes

        # Initialize model after data loaders are potentially set
        self.model = self.initialize_model(len(self.classes))
        self.criterion = fct_loss(choice_loss)
        self.optimizer = choice_optimizer_fct(self.model, choice_optim=choice_optimizer, lr=self.learning_rate, weight_decay=1e-6)
        self.scheduler = choice_scheduler_fct(self.optimizer, choice_scheduler=choice_scheduler, step_size=10, gamma=0.1)
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path

    def initialize_model(self, num_classes=10):
        if self.model_choice in ["LSTM", "GRU"]:
            # Model for time series forecasting
            # Ensure data loaders are set before this point or handle it differently
            n_hidden = 25
            n_layers = 2
            input_dim = next(iter(self.train_loader))[0].shape[2] if self.train_loader else 10  # Default or error
            model = RnnNet(model_choice=self.model_choice, input_size=input_dim, hidden_size=n_hidden,
                           num_layers=n_layers, batch_first=True, device=self.device)

        elif self.model_choice in ["CNNCifar", "mobilenet", "CNNMnist"]:
            # model for a classification problem
            model = Net(num_classes=num_classes, arch=self.model_choice)

        else:
            model = Model()

        return validate_dp_model(model.to(self.device)) if self.dp else model.to(self.device)

    @classmethod
    def node(cls, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        if kwargs['name_dataset'] in ["cifar", "mnist", "alzheimer"]:
            test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        else:
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.test_loader = DataLoader(dataset=test_data, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)
        return obj

    @classmethod
    def client(cls, x_train, y_train, x_val, y_val, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        if kwargs['name_dataset'] in ["cifar", "mnist", "alzheimer"]:
            train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
            val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))
            test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        else:
            train_data = TensorDataset(torch.FloatTensor(x_train).squeeze(-1), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(x_val).squeeze(-1), torch.FloatTensor(y_val))
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.len_train = len(y_train)
        obj.train_loader = DataLoader(dataset=train_data, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)
        obj.val_loader = DataLoader(dataset=val_data, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)
        obj.test_loader = DataLoader(dataset=test_data, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)  # Test loader might not need padding
        return obj

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_dict_params(self, config):
        return {f'param_{i}': val.cpu().numpy() for i, (_, val) in enumerate(self.model.state_dict().items())}

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, node_id, config):
        self.set_parameters(parameters)
        if self.dp:
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=self.epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
        results = train(node_id, self.model, self.train_loader, self.val_loader,
                        self.epochs, self.criterion, self.optimizer, self.scheduler, device=self.device,
                        dp=self.dp, delta=self.delta,
                        max_physical_batch_size=int(self.batch_size / 4), privacy_engine=self.privacy_engine)
        # print(f"Node {node_id} Epoch {e}: Train loss: {epoch_loss}, Train acc: {epoch_acc}, Val loss: {val_loss}, Val acc: {val_acc}")

        # Save results
        if self.save_results:
            save_graphs(self.save_results, self.epochs, results)

        return self.get_parameters(config={}), {'len_train': self.len_train, **results}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy, y_pred, y_true, y_proba = test(self.model, self.test_loader, self.criterion, device=self.device)

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path,  # + f"_client_{self.cid}.png",
                            self.classes)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path,  # + f"_client_{self.cid}.png",
                         len(self.classes))

        return {'test_loss': loss, 'test_acc': accuracy}
