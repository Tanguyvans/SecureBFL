from node import NodeClient, NodeServer
import torch
import numpy as np
from block import Block
from blockchain import Blockchain
import hashlib

if __name__ == "__main__":
    NodeServer1 = NodeServer("1234", "node1")
    NodeClient1 = NodeClient()
    NodeServer2 = NodeServer("2345", "node2")
    NodeClient2 = NodeClient()

    NodeClient1.clientConnection(NodeServer2)
    NodeClient2.clientConnection(NodeServer1)

    block = NodeServer1.MineBlock()
    NodeClient1.broadcast_block(block)

    block = NodeServer2.MineBlock()
    NodeClient2.broadcast_block(block)

    NodeServer1.aggregateParameters()


    # print("===== Node1 =====")
    # NodeServer1.blockchain.show_chain()
    # print("===== Node2 =====")
    # NodeServer2.blockchain.show_chain()

