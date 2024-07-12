from collections import OrderedDict
import numpy as np
import torch
import flwr as fl

from going_modular.model import Net

from going_modular.security import PrivacyEngine, validate_dp_model
from going_modular.data_setup import TensorDataset, DataLoader
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct, save_graphs,
                                 save_matrix, save_roc)
from going_modular.engine import train, test

import os


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, epochs=1, model_choice="simplenet", dp=True, delta=1e-5, epsilon=0.5,
                 max_grad_norm=1.2, name_dataset="cifar", device="gpu", classes=(*range(10),),
                 learning_rate=0.001, choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None,
                 save_results=None, matrix_path=None, roc_path=None, patience=2):
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
        self.patience = patience

        # Initialize model after data loaders are potentially set
        model = Net(num_classes=len(self.classes), arch=self.model_choice)
        self.model = validate_dp_model(model.to(self.device)) if self.dp else model.to(self.device)
        self.criterion = fct_loss(choice_loss)
        self.optimizer = choice_optimizer_fct(self.model, choice_optim=choice_optimizer, lr=self.learning_rate, weight_decay=1e-6)
        #self.scheduler = choice_scheduler_fct(self.optimizer, choice_scheduler=choice_scheduler, step_size=10, gamma=0.1)
        self.scheduler = None
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path

    @classmethod
    def node(cls, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        obj.test_loader = DataLoader(dataset=test_data, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)
        return obj

    @classmethod
    def client(cls, x_train, y_train, x_val, y_val, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
        val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

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
                        max_physical_batch_size=int(self.batch_size / 4), privacy_engine=self.privacy_engine,
                        patience=self.patience)
 
        self.model.load_state_dict(torch.load(f"models/{node_id}_best_model.pth"))
        best_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        self.set_parameters(best_parameters)
        
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
