# DFL Architecture with PBFT Protocol

SecureBFL is a decentralized federated learning framework that leverages the Practical Byzantine Fault Tolerance (PBFT) protocol to ensure secure and reliable model training in a decentralized manner. To ensure the privacy of the model, we use Multi-party Computation (SMPC) technique.

## How it works

The architecture of SecureBFL is illustrated in the following image:

<img src="img/global.png" alt="SecureBFL Architecture" width="500" height="auto">

## Protocol

The Practical Byzantine Fault Tolerance (PBFT) protocol is a consensus algorithm designed for distributed systems. It ensures agreement among nodes even in the presence of faulty or malicious actors. PBFT allows nodes to reach consensus through a series of message exchanges, and it guarantees system integrity as long as fewer than one-third of the nodes are faulty. This makes PBFT a robust choice for applications like decentralized federated learning, providing a reliable method for achieving agreement and consistency in distributed environments.

## Guidelines

To get started with the DFL architecture using the PBFT protocol, follow these simple steps:

### Install Dependencies:

Run the following command to install the required dependencies from the requirements.txt file:

```
pip install -r requirements.txt
```

### Set up the configuration:

You can set up the configuration in the `config.py` file. Here's an overview of some key settings:

- `name_dataset`: Choose between "cifar" or "mnist" datasets.
- `arch`: Select the model architecture ("simpleNet", "CNNCifar", or "mobilenet").
- `pretrained`: Whether to use pretrained weights (True/False).
- `batch_size`: Number of samples per batch during training.
- `n_epochs`: Number of training epochs.
- `number_of_nodes`: Number of nodes in the network.
- `number_of_clients_per_node`: Number of clients connected to each node.
- `min_number_of_clients_in_cluster`: Minimum number of clients required in a cluster.
- `n_rounds`: Number of federated learning rounds.
- `choice_loss`: Loss function to use (e.g., "cross_entropy").
- `choice_optimizer`: Optimization algorithm (e.g., "Adam").
- `lr`: Learning rate for the optimizer.
- `choice_scheduler`: Learning rate scheduler (e.g., "StepLR" or None).
- `diff_privacy`: Whether to use differential privacy (True/False).
- `secret_sharing`: Type of secret sharing scheme ("additif" or "shamir").
- `k` and `m`: Parameters for secret sharing (k-out-of-m scheme).
- `ts`: Time step parameter.

Adjust these settings according to your specific requirements and experimental setup.

### Run the SecureBFL:

Execute the main script main.py to initiate the DFL architecture and start the federated learning process:

```
python main_bfl.py
```

### View Outputs:

After the execution is complete, you can examine the output files:

output.txt: This file contains the overall output and logs from the federated learning process.
node1.txt: Output specific to Node 1.
node2.txt: Output specific to Node 2.
node3.txt: Output specific to Node 3.
Feel free to explore the generated files to gain insights into the decentralized federated learning process.
