import logging

from block import Block
from protocols.consensus_protocol import ConsensusProtocol


class PBFTProtocol(ConsensusProtocol):
    def __init__(self, node, blockchain):
        self.node = node
        self.node_id = self.node.id

        self.prepare_counts = {}
        self.commit_counts = {}
        self.model_usefullness = {}

        self.blockchain = blockchain

    def handle_message(self, message):
        message_type = message.get("type")

        if message_type == "request":
            self.request(message["content"])
            return 

        if message["id"] not in self.node.peers: 
            return  

        public_key = self.node.peers[message["id"]]["public_key"]
        msg = {"type": message["type"], "content": message["content"]}
        is_valid_signature = self.node.verify_signature(message["signature"], msg, public_key)

        if not is_valid_signature: 
            logging.warning("Not valid signature: %s", message)
            return 

        logging.info("Valid signature: %s", is_valid_signature)    

        if message_type == "pre-prepare":
            response = self.pre_prepare(message["content"])
        elif message_type == "prepare":
            response = self.prepare(message["content"])
        elif message_type == "commit":
            response = self.commit(message['id'], message["content"])
        else:
            logging.warning("Unknown message type: %s", message_type)

        return response

    def request(self, content):
        block = self.create_block_from_request(content)
        message = {"type": "pre-prepare", "content": block.to_dict()}
        self.node.broadcast_message(message)

        return "requested"

    def pre_prepare(self, message):
        logging.info("Node %s received pre-prepare for block: \n%s", self.node_id, message)

        block = Block(message["index"], message["model_type"], message["storage_reference"], message["calculated_hash"], message["participants"],
                      message["previous_hash"])

        self.prepare_counts[message["current_hash"]] = 0

        message = {"type": "prepare", "content": block.to_dict()}
        self.node.broadcast_message(message)

        return "pre-prepared"

    def prepare(self, message):
        logging.info("Node %s received prepare for block %s", self.node_id, message)
        block_hash = message["current_hash"]
        if block_hash not in self.prepare_counts: 
            self.prepare_counts[block_hash] = 0
        self.prepare_counts[block_hash] += 1

        if self.is_prepared(block_hash): 
            if message["model_type"] == "update": 
                if message["storage_reference"] not in self.model_usefullness: 
                    self.model_usefullness[message["storage_reference"]] = self.node.is_update_usefull(message["storage_reference"], message["participants"])

                message["usefull"] = self.model_usefullness[message["storage_reference"]]

                if block_hash not in self.commit_counts: 
                    self.commit_counts[block_hash] = {"count": 0, "senders": []}

                if message["usefull"] == True and self.node_id not in self.commit_counts[block_hash]["senders"]:
                    self.commit_counts[block_hash]["count"] += 1
                    self.commit_counts[block_hash]["senders"].append(self.node_id)
            else:
                message["usefull"] = True

                if block_hash not in self.commit_counts: 
                    self.commit_counts[block_hash] = {"count": 0, "senders": []}

                if message["usefull"] == True and self.node_id not in self.commit_counts[block_hash]["senders"]:
                    self.commit_counts[block_hash]["count"] += 1
                    self.commit_counts[block_hash]["senders"].append(self.node_id)

            commit_message = {"type": "commit", "content": message}
            self.node.broadcast_message(commit_message)
            logging.info("Node %s prepared block to %s", self.node_id, self.node.peers)
            return "prepared"
        else:
            logging.info("Node %s waiting for more prepares for block %s", self.node_id, block_hash)
            return "waiting"

    def commit(self, sender, message):
        logging.info("Node %s received commit for block %s", self.node_id, message)
        block_hash = message["current_hash"]

        if message["model_type"] == "update": 
            if block_hash not in self.commit_counts: 
                self.commit_counts[block_hash] = {"count": 0, "senders": []}

            if sender not in self.commit_counts[block_hash]["senders"] and message["usefull"] == True: 
                self.commit_counts[block_hash]["count"] += 1
                self.commit_counts[block_hash]["senders"].append(sender)
        else: 
            if block_hash not in self.commit_counts: 
                self.commit_counts[block_hash] = {"count": 0, "senders": []}

            if sender not in self.commit_counts[block_hash]["senders"]: 
                self.commit_counts[block_hash]["count"] += 1
                self.commit_counts[block_hash]["senders"].append(sender)

        block = Block(
            message["index"], 
            message["model_type"],
            message["storage_reference"], 
            message["calculated_hash"], 
            message["participants"],
            message["previous_hash"]
        )

        is_global_model = message["model_type"] in ["first_global_model", "global_model"]

        with open("results/BFL/output.txt", "a") as file:
            file.write(f"node: {self.node_id} model: {message['storage_reference']} commit_counts: {self.commit_counts[block_hash]} \n")

        if is_global_model or self.can_commit(block_hash):
            logging.info("Node %s committing block %s", self.node_id, block_hash)

            if self.validate_block(message):
                
                # il faut améliorer cette condition
                if self.blockchain.blocks[-1].index + 1 == block.index:
                    self.blockchain.add_block(block)

                if message["model_type"] == "first_global_model" and self.node.global_params_directory == "": 
                    self.node.global_params_directory = message["storage_reference"]

                if message["model_type"] == "global_model": 
                    self.node.global_params_directory = message["storage_reference"]

                logging.info("Node %s committed block %s", self.node_id, block_hash)

                return "added"
            else:
                logging.warning("Invalid block. Discarding commit")
                return "invalid"
        else:
            logging.info("Node %s waiting for more commits for block %s", self.node_id, block_hash)
            return "waiting"

    def validate_block(self, block_data):
        """
        Verify the integrity of the block
        :param block_data:
        :return:
        """

        if ("index" not in block_data or "model_type" not in block_data or "storage_reference" not in block_data
                or "calculated_hash" not in block_data or "previous_hash" not in block_data):
            return False

        # Verify that the index is correctly incremented
        if block_data["index"] != self.blockchain.blocks[-1].index + 1:
            return False

        # Verify the validity of the previous hash
        previous_block = self.blockchain.blocks[-1] if self.blockchain.blocks else None
        if previous_block and block_data["previous_hash"] != previous_block.current_hash:
            return False
        
        return True

        # if block_data["model_type"] == "update": 
        #     if block_data["storage_reference"] not in self.model_usefullness: 
        #         self.model_usefullness[block_data["storage_reference"]] = self.node.is_update_usefull(block_data["storage_reference"], block_data["participants"])

        #     return self.model_usefullness[block_data["storage_reference"]]

        if block_data["model_type"] == "global_model":
            # return self.node.is_global_valid(block_data["calculated_hash"])
            return True

        if block_data["model_type"] == "first_global_model":
            return True
        
        return False

    def create_block_from_request(self, content):
        previous_blocks = self.blockchain.blocks
        index_of_new_block = len(previous_blocks)
        model_type = content.get("model_type")
        storage_reference = content.get("storage_reference")
        calculated_hash = content.get("calculated_hash")
        participants = content.get("participants")
        previous_hash_of_last_block = previous_blocks[-1].current_hash

        # create a new block from the client's request
        new_block = Block(index_of_new_block, model_type, storage_reference, calculated_hash, participants,
                          previous_hash_of_last_block)

        return new_block
    
    def is_prepared(self, id):
        return self.prepare_counts[id] >= 1

    def can_commit(self, id):
        print(self.commit_counts[id])
        return self.commit_counts[id]["count"] >= 2