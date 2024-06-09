import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(22, 5)
        self.layer_out = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        
    def forward(self, inputs, device):
        x = self.relu(self.layer_1(inputs))
        x = self.layer_out(x)

        return x

class RnnNet(nn.Module):
    def __init__(self, model_choice="LSTM", input_size=1, hidden_size=32, num_layers=2, batch_first=True):
        super().__init__()
        if model_choice not in ["LSTM", "GRU"]:
            raise ValueError("Model choice must be 'LSTM' or 'GRU'")
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first) if model_choice == "LSTM" else nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 1)
        self.model_choice = model_choice
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, device):
        # Ensure x has a batch dimension if it's missing
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if self.model_choice == "LSTM":
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            long_term_memory = (h0, c0)
        else:
            long_term_memory = h0

        # Forward pass through RNN
        out, _ = self.model(x.to(device), long_term_memory)
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

    def forward(self, x, device):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
