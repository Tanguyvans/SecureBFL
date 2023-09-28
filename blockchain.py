from block import Block
import numpy as np 
import hashlib

class Blockchain:
    def __init__(self):
        self.blocks = []
        genesis_block = Block.create_genesis_block()
        self.blocks.append(genesis_block)

    def add_block(self, block, hash):
        if self.is_valid_block(block, hash):
            self.blocks.append(block)
            return True
        return False     

    def is_valid_block(self, block, hash):
        loaded_weights_dict = np.load(block.storage_reference)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]
        loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])

        hash_model = hashlib.sha256()
        hash_model.update(str(loaded_weights).encode('utf-8'))
        hash_model = hash_model.hexdigest()

        if block.cryptographic_hash == hash and block.calculated_hash == hash_model: 
            return True
        else: 
            return False

    @property
    def len_chain(self): 
        return len(self.blocks)

    def show_chain(self): 
        for block in self.blocks: 
            print("\n\n================")
            print("prev_hash:\t\t", block.previous_block_cryptographic_hash)
            print("Data:\t\t", block.storage_reference, block.calculated_hash)
            print("Number:\t\t", block.block_number)
            print("Hash:\t\t", block.cryptographic_hash)
            print("\n\n================")