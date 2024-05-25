import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, Accuracy
import torch

from model import RnnNet
import time

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from flowerclient import Data, sMAPE


# %%
def test_function2(net, dataloader_test, scaler, label_scaler, device=torch.device("cpu")):
    """
    Model evaluation on test data
    :param net:
    :param dataloader_test:
    :return:
    """
    # Define MSE metric
    mse = MeanSquaredError()

    net.eval()
    list_outputs = []
    list_targets = []
    with torch.no_grad():
        for seqs, labels in dataloader_test:
            # Move data to device
            seqs, labels = seqs.float().to(device), labels.float().to(device)
            # seqs = seqs.view(*seqs.shape, 1)
            # Pass seqs to net and squeeze the result
            outputs = net(seqs, device)

            if label_scaler:
                outputs = torch.tensor(scaler.inverse_transform(outputs), device=device)
                labels = torch.tensor(label_scaler.inverse_transform(labels), device=device)

            outputs = outputs.squeeze()
            labels = labels.squeeze()

            # Compute loss
            mse(outputs, labels)
            list_targets.append(labels.detach().numpy())
            list_outputs.append(outputs.detach().numpy())

    # Compute final metric value
    test_mse = mse.compute()
    print(f"Test MSE: {test_mse}")

    return np.array(list_outputs), np.array(list_targets), test_mse


def train_function2(net, criterion, optimizer, train_loader, n_epochs=5, device=torch.device("cpu")):
    for epoch in range(n_epochs):
        for seqs, labels in train_loader:
            # Move data to device
            seqs, labels = seqs.float().to(device), labels.float().to(device)

            # Reshape model inputs
            #seqs = seqs.view(*seqs.shape, 1)

            # Get model outputs
            outputs = net(seqs, device=device)

            # Compute loss
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    return net


def data_preparation(filename, name_dataset="Airline Satisfaction", number_of_nodes=3):
    if name_dataset == "Airline Satisfaction":
        # Store csv file in a Pandas DataFrame
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0', 'id'], axis=1)
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
        df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        df = df.dropna()

    elif name_dataset == "Energy":
        df = pd.read_pickle(filename)
        df["ID"] = df["ID"].astype("category")
        df["time_code"] = df["time_code"].astype("uint16")
        df = df.set_index("date_time")

        # Electricity consumption per hour (date with hour in the index)
        df = df["consumption"].resample("60min", label='right', closed='right').sum().to_frame()

    else:
        raise ValueError("Dataset not recognized")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    num_samples = len(df_shuffled)
    split_size = num_samples // number_of_nodes

    multi_df = []
    multi_df_test = []
    for i in range(number_of_nodes):
        if i < number_of_nodes - 1:
            subset = df.iloc[i * split_size:(i + 1) * split_size]

        else:
            subset = df.iloc[i * split_size:]

        if name_dataset == "Airline Satisfaction":
            # x are the features and y the target
            x_s = subset.drop(['satisfaction'], axis=1)
            y_s = subset[['satisfaction']]
            multi_df.append([x_s, y_s])

        elif name_dataset == "Energy":
            series = darts.TimeSeries.from_dataframe(subset, value_cols="consumption")
            train, val = series.split_before(pd.Timestamp("20100714"))
            multi_df.append(train)
            multi_df_test.append(val)

        else:
            raise ValueError("Dataset not recognized")

    return multi_df, multi_df_test if name_dataset == "Energy" else multi_df


def create_sequences(data, seq_length):
    """
    Create sequences of data for training the model
    :param data: the dataframe containing the data or the numpy array containing the data
    :param seq_length: the length of the inputs (window size), so the number of points to consider in one training example
    :return: the numpy arrays of the inputs and the targets
    """
    xs, ys = [], []
    # Iterate over data indices
    for i in range(len(data) - seq_length):
        if type(data) is pd.DataFrame:
            # Define inputs
            x = data.iloc[i:i + seq_length]

            # Define target
            y = data.iloc[i + seq_length]

        else:
            # Define inputs
            x = data[i:i + seq_length]

            # Define target
            y = data[i + seq_length]

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


# %%
if __name__ == '__main__':
    # %%
    data_root = "Data"
    name_dataset = "Energy"  # or Energy
    data_folder = f"{data_root}/{name_dataset}"

    ID = 1239

    numberOfNodes = 3
    numberOfClientsPerNode = 6  # corresponds to the number of clients per node, n in the shamir scheme
    min_number_of_clients_in_cluster = 3


    file_path = f"{data_folder}/Electricity/residential_{ID}.pkl"
    number_of_nodes = numberOfClientsPerNode * numberOfNodes
    starter = pd.Timestamp("20100715")  # pd.Timestamp("20100715") #  pd.Timestamp("19590101")

    # %% read data with a private dataset (Energy)
    df = pd.read_pickle(file_path)
    df["ID"] = df["ID"].astype("category")
    df["time_code"] = df["time_code"].astype("uint16")
    df = df.set_index("date_time")

    # Electricity consumption per hour (date with hour in the index)
    df = df["consumption"].resample("60min", label='right', closed='right').sum().to_frame()

    # %% Define the device
    device = torch.device("mps")

    # %% create TensorDataset with datacamp
    # %% Scaling the input data
    sc = None  # MinMaxScaler()
    label_sc = None  # MinMaxScaler()
    window_size = 7*4  # number of data points used as input to predict the next data point
    data = create_sequences(df["2009-07-15": "2010-07-14"], window_size)
    data_test = create_sequences(df["2010-07-14": "2010-07-21"], window_size)
    # Obtaining the scaler for the labels(usage data) so that output can be
    # re-scaled to actual value during evaluation
    #label_sc.fit(data_test)

    # %% Use create_sequences to create inputs and targets
    train_x, train_y = create_sequences(data if sc else df["2009-07-15": "2010-07-14"], window_size)
    print(train_x.shape, train_y.shape)

    test_x, test_y = create_sequences(data_test if label_sc else df["2010-07-14": "2010-07-21"], window_size)

    # %% Pytorch data loaders/generators
    batch_size = 1024

    # Create TensorDataset
    # train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    train_data = Data(torch.FloatTensor(train_x), torch.FloatTensor(train_y))

    # Drop the last incomplete batch
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, drop_last=True
    )
    del train_x, train_y

    # %% Test data
    # test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())
    test_data = Data(torch.FloatTensor(test_x), torch.FloatTensor(test_y))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    del test_x, test_y
    # %% Parameters for the model
    # seq_len = 90  # (timestamps)
    input_chunk_length = 7 * 24  # Number of past time steps that are fed to the forecasting module at prediction time.
    training_length = 7 * 24  # The length of both input (target and covariates) and output (target) time series used during training (>input_chunk_length)
    n_rnn_layers = 3  # The number of recurrent layers.
    hidden_dim = 25,  # Size for feature maps for each hidden RNN layer (:math:`h_n`).

    n_hidden = 25  # number of features in the hidden state h  # 32
    n_layers = 2  # number of recurrent layers.
    n_epochs = 20
    print_every = 100
    lr = 0.001
    input_dim = next(iter(train_loader))[0].shape[2]  # number of features in the input x
    output_dim = 1  # number of features in the output y

    # %% with LSTM
    # %% Define the model with LSTM
    #model_lstm = LSTMNet(input_dim, n_hidden, output_dim, n_layers).to(device)
    model_lstm = RnnNet(model_choice="LSTM", input_size=input_dim, hidden_size=n_hidden, num_layers=n_layers,
                        batch_first=True).to(device)

    # %% Defining loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=lr)

    # %% Train the model with LSTM
    lstm_model = train_function2(model_lstm, criterion, optimizer, train_loader, n_epochs=n_epochs, device=device)

    # %%Evaluate the LSTM model

    lstm_model.to("cpu")

    list_outputs, list_targets, test_mse = test_function2(lstm_model, test_loader, sc, label_sc, torch.device("cpu"))

    s_mape = round(sMAPE(np.array(list_outputs), np.array(list_targets)), 3)
    print(f"sMAPE: {s_mape}%")

    # %% visualizations
    plt.plot(list_outputs, "-o", color="black", label="LSTM Predictions", markersize=3)
    plt.plot(list_targets, color="yellow", label="Actual")
    plt.ylabel("Energy Consumption (MW)")
    plt.title(f"Energy Consumption for Electricity state")
    plt.legend()
    plt.show()
