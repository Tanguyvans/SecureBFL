import torch.nn as nn


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
        
    def forward(self, inputs):
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
