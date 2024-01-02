import hashlib
import json

class Block: 
    def __init__(self, index, model_type, storage_reference, calculated_hash, previous_hash):
        self.index = index
        self.model_type = model_type
        self.storage_reference = storage_reference
        self.calculated_hash = calculated_hash
        self.previous_hash = previous_hash

    @property
    def current_hash(self):
        # Calculer et renvoyer le hash actuel à chaque accès
        block_string = f"{self.index}{self.model_type}{self.storage_reference}{self.calculated_hash}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __str__(self):
        return f"================\n" \
               f"prev_hash:\t {self.previous_hash}\n" \
               f"index:\t\t {self.index}\n" \
               f"model_type:\t\t {self.model_type}\n" \
               f"storage_reference:\t\t {self.storage_reference}\n" \
               f"calculated_hash:\t\t {self.calculated_hash}\n" \
               f"Hash:\t\t {self.current_hash}\n"
    
    def to_dict(self):
        # Convertissez les attributs du bloc en un dictionnaire
        return {
            "index": self.index,
            "storage_reference": self.storage_reference,
            "model_type": self.model_type,
            "previous_hash": self.previous_hash,
            "calculated_hash": self.calculated_hash,
            "current_hash": self.current_hash
        }