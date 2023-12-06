# Test chain

In this project we are creating a DFL (Decentralised Federated Learning) architecture. In htis architecture we are creating multiple nodes.

## Done

1. Create and use a dataset to train the models. We use the "Airline Satisfaction" dataset from Kaggle with more than 100.000 training data.
2. Use flower to train and aggregate the models.
3. Create multiple nodes and use grpc for communication.
4. Check and add blocks to the chains of each node.
5. Check if the block and models are valid.
6. Generate false data on multiple nodes.
7. Store only the reference hash and the hash of the weights on the model.
8. Creation of the client layer to send the models to the nodes
9. Creation of the cluster among clients
10. Creation of the SMPC protocol among the clients

## ToDo

### Communcation protocol

1. Create a final consensus protocol.
2. The global should be validated by multiple nodes before adding it to the chain.
3. Gossip ? At the moment the model is fully connected
4. Create a warning when the node has poisonned data.
5. client authentification (certificate to verify who they are claiming to be)

### SMPC

1. Check that all the clients have sent their SMPC model before doing the aggregation
2.

## proto

Update proto file
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. client.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node.proto
