import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.hub import load_state_dict_from_url
import torchvision.models as models


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

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
        self.fc3 = nn.Linear(84, num_classes)

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

        # with Batch Normalization layers for the non-iid data
        # self.bn1 = nn.BatchNorm2d(6)

        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)

        # with Batch Normalization layers
        # self.bn2 = nn.BatchNorm2d(16)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # with Batch Normalization layers
        # self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)

        # with Batch Normalization layers
        # self.bn4 = nn.BatchNorm1d(84)

        self.fc3 = nn.Linear(84, num_classes)  # 1 if num_classes <= 2 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.pool(F.relu(self.conv1(x)))  # self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))  # self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten the tensor into a vector
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))  # F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.fc2(x))  # F.relu(self.bn4(self.fc2(x)))

        # output layer
        x = self.fc3(x)
        return x


class EfficientNet(nn.Module):
    """
    A CNN model based on EfficientNet (B0 or B4)
    """
    def __init__(self, num_classes=10, arch="efficientnetB4", pretrained=True):
        super(EfficientNet, self).__init__()

        # ////////////////////////////////// efficientNet (B0 ou B4) ///////////////////////////////////////
        if '0' in arch:
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

        if pretrained:
            self.model = archi(weights="DEFAULT")  # DEFAULT means the best available weights from ImageNet.
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],
                nn.Linear(num_ftrs, int(num_ftrs / 4)),
                nn.Linear(int(num_ftrs / 4), num_classes),  # 1 if num_classes <= 2 else num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.model = archi(weights=None, num_classes=num_classes)

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
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNet, self).__init__()
        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = models.mobilenet_v2(weights='DEFAULT')
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],
                nn.Linear(num_ftrs, int(num_ftrs / 4)),
                nn.Linear(int(num_ftrs / 4), num_classes),  # 1 if num_classes <= 2 else num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.model = models.mobilenet_v2(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        out = self.model.features(x)

        features = out.view(out.size(0), -1)

        out = self.model.classifier(features)

        return (out, features) if return_features else out


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(SqueezeNet, self).__init__()
        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = models.squeezenet1_0(weights='DEFAULT')
            num_ftrs = self.model.classifier[1].in_channels

            self.model.classifier = nn.Sequential(
                self.model.classifier[0],  # nn.Dropout(p=0.5),
                nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1)),  # nn.Conv2d(num_ftrs, num_classes, kernel_size=1),
                self.model.classifier[2],  # nn.ReLU(inplace=True),
                self.model.classifier[3],  # nn.AdaptiveAvgPool2d((1, 1))
                # nn.Softmax(dim=1)
            )
        else:
            self.model = models.squeezenet1_0(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        out = self.model.features(x)
        out = self.model.classifier(out)  # self.model.classifier(features)
        out = torch.flatten(out, 1)

        if return_features:
            features = out.view(out.size(0), -1)
            return out, features

        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, arch="resnet18", pretrained=True):
        super(ResNet, self).__init__()
        if '18' in arch:
            archi = models.resnet18
        elif '34' in arch:
            archi = models.resnet34
        elif '50' in arch:
            archi = models.resnet50
        elif '101' in arch:
            archi = models.resnet101
        elif '152' in arch:
            archi = models.resnet152
        else:
            raise NotImplementedError("The architecture is not implemented")

        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = archi(weights="DEFAULT")
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes, bias=True),
                # nn.Linear(num_ftrs//4, num_ftrs//8, bias=True),
                # nn.Linear(num_ftrs // 8, num_classes, bias=True),
                # nn.Softmax(dim=1)
            )

        else:
            self.model = archi(weights=None, num_classes=num_classes)

    def forward(self, x):
        out = self.model(x)

        return out


# generic class to choose the architecture of the model
class Net(nn.Module):
    """
    This is a generic class to choose the architecture of the model.
    param num_classes: the number of classes
    param arch: the architecture of the model (str)

    return: the model
    """
    def __init__(self, num_classes=10, input_channels=3, arch="simpleNet",  pretrained=True) -> None:
        super(Net, self).__init__()
        print("Number of classes : ", num_classes, " and the architecture is : ", arch)
        if "simplenet" in arch.lower():
            self.model = SimpleNet(num_classes=num_classes)

        elif "efficientnet" in arch.lower():
            self.model = EfficientNet(num_classes=num_classes, arch=arch)

        elif "mobilenet" in arch.lower():
            self.model = MobileNet(num_classes=num_classes)

        elif "cifar" in arch.lower():
            self.model = CNNCifar()

        elif "mnist" in arch.lower():
            self.model = CNNMnist()
        elif "squeeze" in arch.lower():
            self.model = SqueezeNet(num_classes=num_classes)
        elif "resnet" in arch.lower():
            self.model = ResNet(num_classes=num_classes, arch=arch, pretrained=pretrained)

        else:
            raise NotImplementedError("The architecture is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """

        # self.model.forward(x)
        return self.model(x)
