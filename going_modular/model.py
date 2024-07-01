import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.hub import load_state_dict_from_url
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(22, 5)
        self.layer_out = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.layer_out(x)

        return x


class RnnNet(nn.Module):
    """
    RNN model for time series forecasting using LSTM or GRU

    Args:
    - model_choice: choice of RNN model. Default: LSTM
    - input_size: number of features in the input x
    - hidden_size: number of features in the hidden state h
    - num_layers: number of recurrent layers. Default: 2
    - batch_first: if True, then the input and output tensors are provided as (batch, seq, feature). Default: True

    Returns:
    - out: tensor containing the output features h_t from the last layer of the RNN, for each t
    """
    def __init__(self, model_choice="LSTM", input_size=1, hidden_size=32, num_layers=2, batch_first=True, device="cpu"):
        super().__init__()
        # Define RNN layer
        if model_choice == "LSTM":
            model = nn.LSTM

        elif model_choice == "GRU":
            model = nn.GRU
        else:
            raise ValueError("Model choice must be 'LSTM' or 'GRU'")

        self.choice_model = model_choice
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Ensure x has a batch dimension if it's missing
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Initialize short-term memory
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        if self.choice_model == "LSTM":
            # Initialize the long-term memory
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            long_term_memory = (h0, c0)
        else:
            # GRU merges short-term memory and long-term memory in a single hidden state
            long_term_memory = h0

        # Forward pass through RNN
        out, _ = self.model(x.to(self.device), long_term_memory)
        # Handle output
        out = self.fc(out[:, -1, :])  # Assuming you want the last timestep
        return out


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Aplatir les images en un vecteur de 28*28
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class SimpleNetMnist(nn.Module):
    """
    A simple CNN model for MNIST
    """
    def __init__(self, num_classes=10, input_channels=3) -> None:
        super(SimpleNetMnist, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 1 if num_classes <= 2 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor into a vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output layer
        x = self.fc3(x)
        return x


class SimpleNet(nn.Module):
    """
    A simple CNN model
    """
    def __init__(self, num_classes=10) -> None:
        super(SimpleNet, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 1 if num_classes <= 2 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor into a vector
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output layer
        x = self.fc3(x)
        return x


class EfficientNet(nn.Module):
    """
    A CNN model based on EfficientNet (B0 or B4)
    """
    def __init__(self, num_classes=10, arch="efficientnetB4"):
        super(EfficientNet, self).__init__()

        # ////////////////////////////////// efficientNet (B0 ou B4) ///////////////////////////////////////
        if 'O' in arch:
            # problem with the actual version of efficientnet (B0)  weights in PyTorch Hub
            # when this problem will be solved, we can use directly the following line
            # archi = models.efficientnet_b0 if '0' in arch else models.efficientnet_b4
            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)

            models._api.WeightsEnum.get_state_dict = get_state_dict

            archi = models.efficientnet_b0

        elif '4' in arch:
            archi = models.efficientnet_b4

        else:
            raise NotImplementedError("The architecture is not implemented")

        self.model = archi(weights="DEFAULT")  # DEFAULT means the best available weights from ImageNet.
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],
            nn.Linear(num_ftrs, int(num_ftrs / 4)),
            nn.Linear(int(num_ftrs / 4), num_classes),  # 1 if num_classes <= 2 else num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, return_features=False):
        out = self.model.features(x)
        out = self.model.avgpool(out)

        features = out.view(out.size(0), -1)

        out = self.model.classifier(features)

        return (out, features) if return_features else out


class MobileNet(nn.Module):
    """
    A CNN model based on MobileNet (V2)
    """
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],
            nn.Linear(num_ftrs, int(num_ftrs / 4)),
            nn.Linear(int(num_ftrs / 4), num_classes),  # 1 if num_classes <= 2 else num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, return_features=False):
        out = self.model.features(x)

        features = out.view(out.size(0), -1)

        out = self.model.classifier(features)

        return (out, features) if return_features else out


# generic class to choose the architecture of the model
class Net(nn.Module):
    """
    This is a generic class to choose the architecture of the model.
    param num_classes: the number of classes
    param arch: the architecture of the model (str)

    return: the model
    """
    def __init__(self, num_classes=10, input_channels=3, arch="simpleNet") -> None:
        super(Net, self).__init__()
        print("Number of classes : ", num_classes, " and the architecture is : ", arch)
        if arch == "simpleNet":
            self.model = SimpleNet(num_classes=num_classes)

        elif "efficientnet" in arch:
            self.model = EfficientNet(num_classes=num_classes, arch=arch)

        elif "mobilenet" in arch:
            self.model = MobileNet(num_classes=num_classes)

        elif "cifar" in arch.lower():
            self.model = CNNCifar()

        elif "mnist" in arch.lower():
            self.model = CNNMnist()

        else:
            raise NotImplementedError("The architecture is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """

        # self.model.forward(x)
        return self.model(x)
