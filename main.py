from node import NodeClient, NodeServer
import torch
import numpy as np
from block import Block
from blockchain import Blockchain
import hashlib

from ucimlrepo import fetch_ucirepo 
from sklearn.utils import shuffle
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    iris = fetch_ucirepo(id=53) 

    X = iris.data.features
    y = iris.data.targets

    df = pd.concat([X, y], axis=1)
    df['species_category'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    df = shuffle(df)

    num_samples = len(df)
    split_size = num_samples // 3

    subset1 = df.iloc[:split_size]
    subset2 = df.iloc[split_size:2*split_size]
    subset3 = df.iloc[2*split_size:]


    NodeServer1 = NodeServer("1111", "node1", subset1)
    NodeClient1 = NodeClient()

    NodeServer2 = NodeServer("1112", "node2", subset2)
    NodeClient2 = NodeClient()

    NodeServer3 = NodeServer("1113", "node3", subset3)
    NodeClient3 = NodeClient()

    NodeClient1.clientConnection([NodeServer2, NodeServer3])
    NodeClient2.clientConnection([NodeServer1, NodeServer3])
    NodeClient3.clientConnection([NodeServer1, NodeServer2])

    block = NodeServer1.CreateGlobalModel()
    NodeClient1.broadcast_block(block)
    NodeServer1.AppendBlock(block)
    NodeClient1.broadcast_decision()

    for i in range(20):
        block = NodeServer1.UpdateModel()
        msg = NodeClient1.broadcast_block(block)
        print(msg)
        if msg['success']/msg['tot'] >= 1/2: 
            NodeServer1.AppendBlock(block)
            NodeClient1.broadcast_decision()

        block = NodeServer2.UpdateModel()
        msg = NodeClient2.broadcast_block(block)
        print(msg)
        if msg['success']/msg['tot'] >= 1/2:
            NodeServer2.AppendBlock(block)
            NodeClient2.broadcast_decision()

        block = NodeServer3.UpdateModel()
        msg = NodeClient3.broadcast_block(block)
        print(msg)
        if msg['success']/msg['tot'] >= 1/2:
            NodeServer3.AppendBlock(block)
            NodeClient3.broadcast_decision()

        answer = NodeServer1.aggregateParameters()
        if answer["status"] == True:
            NodeClient1.broadcast_block(answer["message"]) 
            NodeServer1.AppendBlock(answer["message"])
            NodeClient1.broadcast_decision()
        else: 
            print(answer["message"])

    print("===== Node1 =====")
    NodeServer1.blockchain.save_chain_in_file("node1")
    print("===== Node2 =====")
    NodeServer2.blockchain.save_chain_in_file("node2")
    print("===== Node3 =====")
    NodeServer3.blockchain.save_chain_in_file("node3")

