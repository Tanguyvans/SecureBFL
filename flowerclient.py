from collections import OrderedDict
import numpy as np
import torch
import flwr as fl

from going_modular.model import Model, RnnNet, Net

from going_modular.security import PrivacyEngine, validate_dp_model, BatchMemoryManager
from going_modular.data_setup import TensorDataset, DataLoader
# from pytorch_lightning.callbacks import EarlyStopping


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / len(y_test)  # correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


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


def fct_loss(choice_loss):
    """
    A function to choose the loss function

    :param choice_loss: the choice of the loss function
    :return: the loss function
    """
    choice_loss = choice_loss.lower()
    print("choice_loss : ", choice_loss)

    if choice_loss == "cross_entropy":
        loss = torch.nn.CrossEntropyLoss()

    elif choice_loss == "binary_cross_entropy":
        loss = torch.nn.BCELoss()

    elif choice_loss == "bce_with_logits":
        loss = torch.nn.BCEWithLogitsLoss()

    elif choice_loss == "smape":
        loss = sMAPE()

    elif choice_loss == "mse":
        loss = torch.nn.MSELoss(reduction='mean')

    elif choice_loss == "scratch":
        return None

    else:
        print("Warning problem : unspecified loss function")
        return None

    print("loss : ", loss)

    return loss


def choice_optimizer_fct(model, choice_optim="Adam", lr=0.001, momentum=0.9, weight_decay=1e-6):
    """
    A function to choose the optimizer

    :param model: the model
    :param choice_optim: the choice of the optimizer
    :param lr: the learning rate
    :param momentum: the momentum
    :param weight_decay: the weight decay
    :return: the optimizer
    """

    choice_optim = choice_optim.lower()

    if choice_optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    elif choice_optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad='adam' == 'amsgrad',)

    elif choice_optim == "rmsprop":
        print("oui")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        print("Warning problem : unspecified optimizer")
        return None

    return optimizer


def choice_scheduler_fct(optimizer, choice_scheduler=None, step_size=30, gamma=0.1):
    """
    A function to choose the scheduler

    :param optimizer: the optimizer
    :param choice_scheduler: the choice of the scheduler
    :param step_size: the step size
    :param gamma: the gamma
    :return: the scheduler
    """

    if choice_scheduler:
        choice_scheduler = choice_scheduler.lower()

    if choice_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif choice_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif choice_scheduler == "cycliclr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)

    elif choice_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=0)

    elif choice_scheduler == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=gamma)

    elif choice_scheduler == "plateaulr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08, verbose=False)

    else:
        print("Warning problem : unspecified scheduler")
        return None

    print("scheduler : ", scheduler)
    return scheduler


def choice_device(device):
    """
        A function to choose the device

        :param device: the device to choose (cpu, gpu or mps)
        """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    print("The device is : ", device)
    return device


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, model_choice="simplenet", diff_privacy=True, delta=1e-5, epsilon=0.5,
                 max_grad_norm=1.2, name_dataset="Airline Satisfaction", device="gpu", choice_loss="cross_entropy",
                 num_classes=10):
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
        self.device = choice_device(device)

        # Initialize model after data loaders are potentially set
        self.model = self.initialize_model(choice_loss, num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    def initialize_model(self, choice_loss="cross_entropy", num_classes=10):
        if self.model_choice in ["LSTM", "GRU"]:
            # Model for time series forecasting
            # Ensure data loaders are set before this point or handle it differently
            n_hidden = 25
            n_layers = 2
            input_dim = next(iter(self.train_loader))[0].shape[2] if self.train_loader else 10  # Default or error
            model = RnnNet(model_choice=self.model_choice, input_size=input_dim, hidden_size=n_hidden,
                           num_layers=n_layers, batch_first=True, device=self.device)
            self.criterion = fct_loss(choice_loss)  # nn.MSELoss()

        elif self.model_choice in ["CNNCifar", "mobilenet", "CNNMnist"]:
            # model for a classification problem
            model = Net(num_classes=num_classes, arch=self.model_choice)
            self.criterion = fct_loss(choice_loss)  # nn.CrossEntropyLoss()

        else:
            model = Model()
            self.criterion = fct_loss(choice_loss)  # nn.BCEWithLogitsLoss()

        return validate_dp_model(model.to(self.device)) if self.diff_privacy else model.to(self.device)

    @classmethod
    def node(cls, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        if kwargs['name_dataset'] in ["cifar", "mnist", "alzheimer"]:
            #TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))
            test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        else:
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.test_loader = DataLoader(dataset=test_data, batch_size=1)
        return obj

    @classmethod
    def client(cls, x_train, y_train, x_val, y_val, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        if kwargs['name_dataset'] in ["cifar", "mnist", "alzheimer"]:
            #train_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_train]), torch.tensor(y_train))
            #val_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_val]), torch.tensor(y_val))
            #test_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))

            train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
            val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))
            test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        else:
            train_data = TensorDataset(torch.FloatTensor(x_train).squeeze(-1), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(x_val).squeeze(-1), torch.FloatTensor(y_val))
            test_data = TensorDataset(torch.FloatTensor(x_test).squeeze(-1), torch.FloatTensor(y_test))

        obj.len_train = len(y_train)
        obj.train_loader = DataLoader(dataset=train_data, batch_size=kwargs['batch_size'], shuffle=True)
        obj.val_loader = DataLoader(dataset=val_data, batch_size=kwargs['batch_size'], shuffle=True)
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
            x_batch, y_batch = x_batch.float().to(self.device), y_batch.float().to(self.device)
            y_pred = self.model(x_batch)
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
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(dataloader)
        test_acc = 100 * correct / total
        """
        with torch.inference_mode():  # or with torch.no_grad():
            for x_test_batch, y_test_batch in dataloader:
                # Move data to device and set to float
                x_test_batch, y_test_batch = x_test_batch.float().to(device), y_test_batch.float().to(device)
                # Get model outputs
                y_test_pred = self.model(x_test_batch, device=device)
                if isinstance(self.model, RnnNet):
                    y_test_pred = y_test_pred.squeeze()
                    y_test_batch = y_test_batch.squeeze()
                    if label_scaler:
                        y_test_pred = torch.tensor(scaler.inverse_transform(y_test_pred), device=device)
                        y_test_batch = torch.tensor(label_scaler.inverse_transform(y_test_batch), device=device)

                    print(y_test_batch, y_test_pred)
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

        return list_outputs, list_targets, test_loss, test_acc
        """
        return test_loss, test_acc
