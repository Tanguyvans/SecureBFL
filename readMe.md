# DFL Architecture with PBFT Protocol

SecureBFL is a decentralized federated learning framework that leverages the Practical Byzantine Fault Tolerance (PBFT) protocol to ensure secure and reliable model training in a decentralized manner. To ensure the privacy of the model, we use Multi-party Computation (SMPC) technique.

## How it works

The architecture of SecureBFL is illustrated in the following image:

![SecureBFL Architecture](img/global.png)

## Structure

## Protocol

The Practical Byzantine Fault Tolerance (PBFT) protocol is a consensus algorithm designed for distributed systems. It ensures agreement among nodes even in the presence of faulty or malicious actors. PBFT allows nodes to reach consensus through a series of message exchanges, and it guarantees system integrity as long as fewer than one-third of the nodes are faulty. This makes PBFT a robust choice for applications like decentralized federated learning, providing a reliable method for achieving agreement and consistency in distributed environments.

## Guidelines

To get started with the DFL architecture using the PBFT protocol, follow these simple steps:

### Install Dependencies:

Run the following command to install the required dependencies from the requirements.txt file:

pip install -r requirements.txt

### TODO

- [ ] Applify the evaluation metrics (only loss is implemented)
- [ ] Find and use another dataset
- [ ] Apply other smpc techniques

### Run the Main Script:

Execute the main script main.py to initiate the DFL architecture and start the federated learning process:

python main.py

### View Outputs:

After the execution is complete, you can examine the output files:

output.txt: This file contains the overall output and logs from the federated learning process.
node1.txt: Output specific to Node 1.
node2.txt: Output specific to Node 2.
node3.txt: Output specific to Node 3.
Feel free to explore the generated files to gain insights into the decentralized federated learning process.
