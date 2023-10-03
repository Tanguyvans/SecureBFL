from node import NodeClient, NodeServer
import torch
import numpy as np
from block import Block
from blockchain import Blockchain
import hashlib

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    NodeServer1 = NodeServer("1234", "node1")
    NodeClient1 = NodeClient()

    NodeServer2 = NodeServer("2345", "node2")
    NodeClient2 = NodeClient()

    NodeServer3 = NodeServer("3456", "node3")
    NodeClient3 = NodeClient()

    NodeClient1.clientConnection(NodeServer2)
    NodeClient1.clientConnection(NodeServer3)

    NodeClient2.clientConnection(NodeServer1)
    NodeClient2.clientConnection(NodeServer3)

    NodeClient3.clientConnection(NodeServer1)
    NodeClient3.clientConnection(NodeServer2)

    block = NodeServer1.CreateGlobalModel()
    NodeClient1.broadcast_block(block)
    NodeServer1.AppendBlock(block)
    NodeClient1.broadcast_decision()

    for i in range(10):
        block = NodeServer1.UpdateModel()
        msg = NodeClient1.broadcast_block(block)
        if msg['success']/msg['tot'] >= 2/3: 
            NodeServer1.AppendBlock(block)
            NodeClient1.broadcast_decision()

        block = NodeServer2.UpdateModel()
        NodeClient2.broadcast_block(block)
        if msg['success']/msg['tot'] >= 2/3:
            NodeServer2.AppendBlock(block)
            NodeClient2.broadcast_decision()

        block = NodeServer3.UpdateModel()
        NodeClient3.broadcast_block(block)
        if msg['success']/msg['tot'] >= 2/3:
            NodeServer3.AppendBlock(block)
            NodeClient3.broadcast_decision()

        block = NodeServer1.aggregateParameters()
        NodeClient1.broadcast_block(block) 
        NodeServer1.AppendBlock(block)
        NodeClient1.broadcast_decision()

    print("===== Node1 =====")
    NodeServer1.blockchain.save_chain_in_file("node1")
    print("===== Node2 =====")
    NodeServer2.blockchain.save_chain_in_file("node2")
    print("===== Node3 =====")
    NodeServer3.blockchain.save_chain_in_file("node3")

