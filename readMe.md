# DFL Architecture with PBFT Protocol

Welcome to the Decentralized Federated Learning (DFL) project! This endeavor focuses on redefining machine learning through a decentralized approach. By creating a network of multiple nodes, we enable collaborative model training with a strong emphasis on security and reliability, thanks to the integration of the Practical Byzantine Fault Tolerance (PBFT) protocol.

## Protocol

The Practical Byzantine Fault Tolerance (PBFT) protocol is a consensus algorithm designed for distributed systems. It ensures agreement among nodes even in the presence of faulty or malicious actors. PBFT allows nodes to reach consensus through a series of message exchanges, and it guarantees system integrity as long as fewer than one-third of the nodes are faulty. This makes PBFT a robust choice for applications like decentralized federated learning, providing a reliable method for achieving agreement and consistency in distributed environments.

## Guidelines

To get started with the DFL architecture using the PBFT protocol, follow these simple steps:

### Install Dependencies:

Run the following command to install the required dependencies from the requirements.txt file:

pip install -r requirements.txt

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
