import grpc
import pickle
import random
import concurrent.futures
from sklearn.model_selection import train_test_split
import numpy as np

import client_pb2
import client_pb2_grpc
import node_pb2
import node_pb2_grpc
from flowerclient import FlowerClient

def encrypt(x, n_shares=2):
    shares = [random.uniform(-5, 5) for _ in range(n_shares - 1)]
    shares.append(x - sum(shares))
    return tuple(shares)

# Fonction pour appliquer le SMPC à une liste de listes
def apply_smpc(input_list, n_shares=2):
    encrypted_list = [[] for i in range(n_shares)]
    for inner_list in input_list:
        if isinstance(inner_list[0], np.ndarray):
            encrypted_inner_list = [[] for i in range(n_shares)]
            for inner_inner_list in inner_list: 
                encrypted_inner_inner_list = [[] for i in range(n_shares)]
                for x in inner_inner_list: 
                    crypted_tuple = encrypt(x, n_shares)
                    for i in range(n_shares):
                        encrypted_inner_inner_list[i].append(crypted_tuple[i])

                for i in range(n_shares):
                    encrypted_inner_list[i].append(encrypted_inner_inner_list[i])
        else: 
            encrypted_inner_list = [[] for i in range(n_shares)]
            for x in inner_list: 
                crypted_tuple = encrypt(x, n_shares)


                for i in range(n_shares):
                    encrypted_inner_list[i].append(crypted_tuple[i])
        
        for i in range(n_shares):
            encrypted_list[i].append(np.array(encrypted_inner_list[i]))

    return encrypted_list
# Fonction pour décrypter une liste de listes chiffrées
def decrypt_list_of_lists(encrypted_list):
    decrypted_list = []
    n_shares = len(encrypted_list)

    for i in range(len(encrypted_list[0])): 
        sum_array = np.add(encrypted_list[0][i], encrypted_list[1][i])
        for j in range(2, n_shares): 
            sum_array = np.add(sum_array, encrypted_list[j][i])

        decrypted_list.append(sum_array)

    return decrypted_list

    result = [[] for i in range(len(encrypted_list[0]))]

    for i in range(len(encrypted_list[0])):
        for j in range(len(encrypted_list[0][i])):
            result[i].append(sum(encrypted_list[k][i][j] for k in range(len(encrypted_list))))

    print('## frag')
    for ri in result: 
        print("frai")
        print(len(ri), ri[0])

    return result

class ClientServer(client_pb2_grpc.ClientServiceServicer): 
    def __init__(self, port, id, batch_size, train, test):
        self.id = id
        self.port = port
        self.cluster = 0

        self.frag_weights = []
        self.sum_dataset_number = 0

        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
        client_pb2_grpc.add_ClientServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'127.0.0.1:{self.port}')
        self.server.start()

        X_train, y_train = train
        X_test, y_test = test
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

        self.flower_client = FlowerClient(batch_size, X_train, X_val ,X_test, y_train, y_val, y_test)

    def sendSmpc(self, request, context):
        weights = pickle.loads(request.value)
        self.frag_weights.append(weights)

        self.sum_dataset_number = request.dataset_number

        return client_pb2.SmpcResponse(success=True)
    
    def sendGlobalModel(self, request, context): 
        self.globalWeights = pickle.loads(request.value)
        self.flower_client.set_parameters(self.globalWeights)
        loss = self.flower_client.evaluate(self.globalWeights, {})[0]

        return client_pb2.GlobalModelResponse(success=True)

    def train(self): 
        old_params = self.flower_client.get_parameters({})
        res = old_params[:]
        for i in range(2):
            res = self.flower_client.fit(res, {})[0]
            loss = self.flower_client.evaluate(res, {})[0]
            with open('output.txt', 'a') as f: 
                f.write(f"loss: {loss} \n")

        encripted_lists = apply_smpc(res, self.cluster)
        self.frag_weights.append(encripted_lists.pop())

        return encripted_lists

        # TODO appliquer la méthode de SMPC
        # seg = [[] for i in range(self.cluster)]
        # for ri in res: 
        #     for i in range(self.cluster): 
        #         seg[i].append(ri*0)

        # self.seg = seg[:]
        # self.sum_weights = res

        # return self.seg[:]
        self.sum_weights = res
        return res

    @property
    def sum_weights(self): 
        return decrypt_list_of_lists(self.frag_weights)

class ClientClient: 
    def __init__(self, clientServer):
        self.connections = {}
        self.nodeServer = ""
        self.server = clientServer

    def clientConnection(self, servers): 
        for server in servers: 
            if self.server != server: 
                channel = grpc.insecure_channel(f'127.0.0.1:{server.port}') 
                stub = client_pb2_grpc.ClientServiceStub(channel)
                self.connections[server.id] = stub 

    def clientNodeConnection(self, nodeServer): 
        channel = grpc.insecure_channel(f'127.0.0.1:{nodeServer.port}') 
        stub = node_pb2_grpc.NodeServiceStub(channel)
        self.nodeServer = stub 

    def sendFragmentedWeightsToClients(self, frag_weights): 
        i = 0
        for k, v in self.connections.items(): 
            serialized_data = pickle.dumps(frag_weights[i])
            message = client_pb2.SmpcMessage()
            message.value = serialized_data
            message.dataset_number = 1

            response = v.sendSmpc(message)
            i+= 1

    def sendWeightsToNode(self, weights, len_dataset): 
        serialized_data = pickle.dumps(weights)

        message = node_pb2.ClientMessage()
        message.value = serialized_data
        message.dataset_number = len_dataset
        message.client_id = str(self.server.id)
        
        response = self.nodeServer.AddWeightsFromClient(message)
        print(response)
