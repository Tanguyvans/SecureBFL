from node import NodeClient, NodeServer
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def dataPreparation(filename, numberOfNodes=3): 
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Customer Type'] = df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
    df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
    df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
    df = df.dropna()

    num_samples = len(df)
    split_size = num_samples // numberOfNodes

    subset1 = df.iloc[:split_size]
    subset2 = df.iloc[split_size:2*split_size]
    subset3 = df.iloc[2*split_size:]

    X_s1=subset1.drop(['satisfaction'],axis=1)
    y_s1=subset1[['satisfaction']]

    X_s2=subset2.drop(['satisfaction'],axis=1)
    y_s2=subset2[['satisfaction']]

    X_s3=subset3.drop(['satisfaction'],axis=1)
    y_s3=subset3[['satisfaction']]

    return [(X_s1, y_s1), (X_s2, y_s2), (X_s3, y_s3)]

if __name__ == "__main__":

    with open("output.txt", "w") as f: 
            f.write("")

    train_sets = dataPreparation('Airline Satisfaction/train.csv')
    test_sets = dataPreparation('Airline Satisfaction/test.csv')

    NodeServer1 = NodeServer("1111", "node1", 256, train_sets[0], test_sets[0])
    NodeClient1 = NodeClient()

    NodeServer2 = NodeServer("1112", "node2", 256, train_sets[1], test_sets[1])
    NodeClient2 = NodeClient()

    NodeServer3 = NodeServer("1113", "node3", 256, train_sets[2], test_sets[2])
    NodeClient3 = NodeClient()

    NodeClient1.clientConnection([NodeServer2, NodeServer3])
    NodeClient2.clientConnection([NodeServer1, NodeServer3])
    NodeClient3.clientConnection([NodeServer1, NodeServer2])

    block = NodeServer1.CreateGlobalModel()
    NodeClient1.broadcast_block(block)
    NodeServer1.AppendBlock(block)
    NodeClient1.broadcast_decision()

    for i in range(20):
        server_msg, block = NodeServer1.UpdateModel()
        msg = NodeClient1.broadcast_block(block)
        msg['tot']+=1
        if server_msg == True:
            msg['success']+=1
        else: 
            msg['failure']+=1
        print(msg)
        if msg['success']/msg['tot'] >= 2/3: 
            NodeServer1.AppendBlock(block)
            NodeClient1.broadcast_decision()

        server_msg, block = NodeServer2.UpdateModel()
        msg = NodeClient2.broadcast_block(block)
        msg['tot']+=1
        if server_msg == True:
            msg['success']+=1
        else: 
            msg['failure']+=1
        print(msg)
        if msg['success']/msg['tot'] >= 2/3:
            NodeServer2.AppendBlock(block)
            NodeClient2.broadcast_decision()

        server_msg, block = NodeServer3.UpdateModel()
        msg = NodeClient3.broadcast_block(block)
        msg['tot']+=1
        if server_msg == True:
            msg['success']+=1
        else: 
            msg['failure']+=1
        print(msg)
        if msg['success']/msg['tot'] >= 2/3:
            NodeServer3.AppendBlock(block)
            NodeClient3.broadcast_decision()

        answer = NodeServer1.aggregateParameters()
        if answer["status"] == True:
            msg = NodeClient1.broadcast_block(answer["message"]) 
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

