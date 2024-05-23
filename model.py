import torch.nn as nn
import torch.nn.init as init
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Number of input features is 12.
        # self.layer_1 = nn.Linear(22, 120) 
        # self.layer_2 = nn.Linear(120, 120)
        # self.layer_3 = nn.Linear(120, 120)
        # self.layer_4 = nn.Linear(120, 120)
        # self.layer_5 = nn.Linear(120, 120)
        # self.layer_6 = nn.Linear(120, 120)
        # self.layer_7 = nn.Linear(120, 120)
        # self.layer_out = nn.Linear(120, 1) 

        self.layer_1 = nn.Linear(22, 5)
        self.layer_out = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        
    def forward(self, inputs, device):
        # x = self.relu(self.layer_1(inputs))
        # x = self.relu(self.layer_2(x))
        # x = self.relu(self.layer_3(x))
        # x = self.relu(self.layer_4(x))
        # x = self.relu(self.layer_5(x))
        # x = self.relu(self.layer_6(x))
        # x = self.relu(self.layer_7(x))
        # x = self.layer_out(x)

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
    def __init__(self, model_choice="LSTM", input_size=1, hidden_size=32, num_layers=2, batch_first=True):
        super().__init__()
        # Define RNN layer
        if model_choice == "LSTM":
            model = nn.LSTM

        elif model_choice == "GRU":
            model = nn.GRU
        else:
            raise ValueError("Model choice must be 'LSTM' or 'GRU'")

        self.choice_model = model_choice
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = model(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, device):
        # Initialize short-term memory
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if self.choice_model == "LSTM":
            # Initialize the long-term memory
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            long_term_memory = (h0, c0)

        elif self.choice_model == "GRU":
            # GRU merges short-term memory and long-term memory in a single hidden state
            long_term_memory = h0

        else:
            raise ValueError("Model choice must be 'LSTM' or 'GRU'")

        # Pass all inputs to RNN layer
        out, _ = self.rnn(x, long_term_memory)
        out = self.fc(out[:, -1, :])
        return out
