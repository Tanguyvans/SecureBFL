import hashlib
import json

class Block: 
    def __init__(self, calculated_hash, storage_reference, model_type, previous_block=None):
        self.nonce = "1"
        self.previous_block = previous_block
        self.calculated_hash = calculated_hash
        self.storage_reference = storage_reference
        self.model_type = model_type
        if previous_block: 
            self.block_number = previous_block.block_number+1
        else: 
            self.block_number = 1

    @property
    def previous_block_cryptographic_hash(self):
        previous_block_cryptographic_hash = ""
        if self.previous_block:
            previous_block_cryptographic_hash = self.previous_block.cryptographic_hash
        return previous_block_cryptographic_hash

    @property
    def cryptographic_hash(self) -> str:

        hash = hashlib.sha256()

        block_content = {
            "nonce": str(1),
            "previous_hash": self.previous_block_cryptographic_hash,
            "calculated_hash": self.calculated_hash,
            "model_type": self.model_type,
            "storage_reference": self.storage_reference,
            "block_number": self.block_number
        }

        block_content_bytes = json.dumps(block_content, indent=2).encode('utf-8')
        hash.update(block_content_bytes)

        return hash.hexdigest()
    
    @classmethod
    def create_genesis_block(cls):
        # Créez un bloc "genesis" sans précédent (premier bloc de la chaîne)
        return cls(calculated_hash="", storage_reference="", model_type="")
    
    def show_last_block(self): 
        print("\n\n================")
        print("prev_hash:\t\t", self.previous_block_cryptographic_hash)
        print("Data:\t\t", self.storage_reference, self.calculated_hash)
        print("Model type:\t\t", self.model_type)
        print("Number:\t\t", self.block_number)
        print("Hash:\t\t", self.cryptographic_hash)
        print("\n\n================")


