import grpc
import block_pb2 as pb2
import block_pb2_grpc as pb2_grpc
import concurrent.futures
import hashlib
import numpy as np
from block import Block
from blockchain import Blockchain
from flowerclient import FlowerClient

from flwr.server.strategy.aggregate import aggregate
from sklearn.model_selection import train_test_split
import torch

class NodeServer(pb2_grpc.NodeServiceServicer):
    def __init__(self, port, id, batch_size, train, test):
        self.id = id
        self.port = port
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'127.0.0.1:{self.port}')
        self.server.start()

        X_train, y_train = train
        X_test, y_test = test
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

        self.blockchain = Blockchain()
        self.flower_client = FlowerClient(batch_size, X_train, X_val ,X_test, y_train, y_val, y_test)
        self.params_directories = []

    def MineBlock(self, filename, model_type): 
        loaded_weights_dict = np.load(filename)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])

        hash_model = hashlib.sha256()
        hash_model.update(str(loaded_weights).encode('utf-8'))
        hash_model = hash_model.hexdigest()

        block = Block(calculated_hash=hash_model, storage_reference=filename, model_type=model_type, previous_block=self.blockchain.blocks[-1])

        return block
    
    def AppendBlock(self, block): 
        self.blockchain.add_block(block, block.cryptographic_hash)
        if block.model_type == "global_model": 
            self.global_params_directory = block.storage_reference
        else:
            self.params_directories.append(block.storage_reference)

    def CreateGlobalModel(self): 
        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        model_type = "global_model"

        filename = f"models/m{self.blockchain.len_chain}.npz"
        self.global_params_directory = filename

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        return block
    
    def UpdateModel(self):
        loaded_weights_dict = np.load(self.global_params_directory)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        self.flower_client.set_parameters(loaded_weights)

        old_params = self.flower_client.get_parameters({})
        len_dataset = self.flower_client.fit(old_params, {})[1]

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        model_type = "update"

        filename = f"models/m{self.blockchain.len_chain}.npz"

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        return self.isModelUpdateUsefull(block.storage_reference), block

    def AddBlockRequest(self, request, context): 
        new_block = Block(request.calculated_hash, request.storage_reference, request.model_type, self.blockchain.blocks[-1])
        if self.blockchain.is_valid_block(new_block, request.hash): 
            self.block = new_block
            if request.model_type == "global_model":
                return pb2.Response(success=True, message=f"Block {request.block_number} added to the blockchain successfully {request.hash}")
            else: 
                if self.isModelUpdateUsefull(new_block.storage_reference):
                    print("Usefull")
                    return pb2.Response(success=True, message=f"Block {request.block_number} added to the blockchain successfully {request.hash}")
                else: 
                    print("Not usefull")
                    return pb2.Response(success=False, message=f"Block {request.block_number} was not added, loss too high")
        else: 
            return pb2.Response(success=False, message=f"Block {request.block_number} was not added for wrong hash")
        
    def AddBlockToChain(self, request, constext):
        self.blockchain.add_block(self.block, self.block.cryptographic_hash)
        if self.block.model_type == "global_model": 
            self.global_params_directory = self.block.storage_reference
        else:
            self.params_directories.append(self.block.storage_reference)

        return pb2.Response(success=True)
    
    def aggregateParameters(self):

        print("aggregation, ", self.params_directories)
        if len(self.params_directories) < 2:
            return {"status": False, "message": f"not enough trained models: {len(self.params_directories)}"}
         
        params_list = []
        for cnt, model_directory in enumerate(self.params_directories):
            loaded_weights_dict = np.load(model_directory)
            loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

            loss = self.flower_client.evaluate(loaded_weights, {})[0]
            acc = self.flower_client.evaluate(loaded_weights, {})[2]['accuracy']

            print(f"model{cnt}: loss: {loss}, acc: {acc}")
            with open("output.txt", "a") as f: 
                f.write(f'Model{cnt}: | Test Loss: {loss:.5f} | Test Acc: {acc:.3f} \n')
            loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])
            params_list.append(loaded_weights)

        self.aggregated_params = aggregate(params_list)
        self.flower_client.set_parameters(self.aggregated_params)

        final_loss = self.flower_client.evaluate(self.aggregated_params, {})[0]
        final_acc = self.flower_client.evaluate(self.aggregated_params, {})[2]['accuracy']
        print(f"final_model: loss: {final_loss}, acc: {final_acc}")
        with open("output.txt", "a") as f: 
            f.write(f"final_model: loss: {final_loss}, acc: {final_acc} \n")
        

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 40
        model_type = "global_model"

        filename = f"models/m{self.blockchain.len_chain}.npz"
        self.global_params_directory = filename

        with open(filename, "wb") as f:
            np.savez(f, **weights_dict)

        block = self.MineBlock(filename, model_type)

        self.params_directories = []

        return {"status": True, "message": block} 

    def isModelUpdateUsefull(self, model_directory): 
        print(self.evaluateModel(model_directory), self.evaluateModel(self.global_params_directory)*1.02)
        if self.evaluateModel(model_directory) <= self.evaluateModel(self.global_params_directory)*1.02: 
            return True
        else: 
            return False

    def evaluateModel(self, model_directory):
        loaded_weights_dict = np.load(model_directory)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loss = self.flower_client.evaluate(loaded_weights, {})[0]

        return loss

class NodeClient:
    def __init__(self):
        self.connections = {}

    def clientConnection(self, servers): 
        for server in servers: 
            channel = grpc.insecure_channel(f'127.0.0.1:{server.port}') 
            stub = pb2_grpc.NodeServiceStub(channel)
            self.connections[server.id] = stub 
  
    def broadcast_block(self, block):
        block_message = pb2.BlockMessage()
        block_message.nonce = block.nonce
        block_message.previous_hash = block.previous_block_cryptographic_hash
        block_message.hash = block.cryptographic_hash
        block_message.model_type = block.model_type
        block_message.calculated_hash = block.calculated_hash
        block_message.storage_reference = block.storage_reference
        block_message.block_number = block.block_number
        
        responses = {'success': 0, 'failure': 0, 'tot': 0}
        for k, v in self.connections.items(): 
            response = v.AddBlockRequest(block_message)

            if response.success == True:
                responses['success'] += 1
            else:
                responses['failure'] += 1
            responses['tot'] += 1

        return responses

    def broadcast_decision(self):
        block_message = pb2.BlockValidation()
        block_message.valid = True

        responses = {'success': 0, 'failure': 0, 'tot': 0}
        for k, v in self.connections.items(): 
            response = v.AddBlockToChain(block_message)

            if response.success == True:
                responses['success'] += 1
            else:
                responses['failure'] += 1
            responses['tot'] += 1

        return responses
    
if __name__ == "__main__":
    NodeServer1 = NodeServer("1234", "node1")
    NodeClient1 = NodeClient()
    NodeServer2 = NodeServer("2345", "node2")
    NodeClient2 = NodeClient()

    NodeClient1.clientConnection(NodeServer2)
    NodeClient2.clientConnection(NodeServer1)
