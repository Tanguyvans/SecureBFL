import threading
import time
import logging
import random

from block import Block

class RaftProtocol:
    def __init__(self, node, blockchain):
        self.node = node
        self.node_id = self.node.node_id
        self.total_nodes = len(self.node.peers)
        self.current_term = 0
        self.voted_for = None
        self.commit_index = 0
        self.last_applied = 0
        self.state = "follower"
        self.votes_received = 0
        self.log = []

        self.leader_id = None

        self.prepare_counts = {}
        self.commit_counts = {}

        self.blockchain = blockchain

        self.reset_election_timeout()

    def handle_message(self, message):
        message_type = message.get("type")

        if message_type == "request_vote":
            print("inside the request form")
            self.handle_request_vote(message)
        elif message_type == "vote_response":
            print("Inside the vote response")
            self.handle_vote_response(message)
        elif message_type == "append_entries":
            self.handle_append_entries(message)
        elif message_type == "client_request":
            self.handle_client_request(message)
        elif message_type == "sync_blockchain":
            print(message)
            self.handle_sync_blockchain(message)
        elif message_type == "sync_request":
            print("received sync request")
            self.handle_sync_request(message)

    def handle_request_vote(self, message):
        candidate_id = message.get("candidate_id")
        candidate_term = message.get("candidate_term")

        if candidate_term < self.current_term:
            # Reject the vote if the candidate's term is outdated
            return

        if candidate_term > self.current_term or (candidate_term == self.current_term and not self.voted_for):
            # Vote for the candidate if its term is equal or greater, and the node has not voted yet
            self.voted_for = candidate_id
            self.current_term = candidate_term
            response = {"type": "vote_response", "vote_granted": True, "term": self.current_term}

            self.reset_election_timeout()
        else:
            # Deny the vote
            response = {"type": "vote_response", "vote_granted": False, "term": self.current_term}

        self.node.send_message(candidate_id, response)

    def handle_append_entries(self, message):
        self.leader_id = message.get("leader_id")
        leader_term = message.get("leader_term")
        entries = message.get("entries")

        # Update the current term if the leader has a higher term
        if leader_term > self.current_term:
            self.current_term = leader_term
            self.voted_for = None
            self.state = "follower"
            print(f"Node {self.node_id} updated to follower due to higher term {leader_term}")

        # Reset election timeout to avoid becoming a candidate
        self.reset_election_timeout()

        # Process the log entries from the leader
        if entries:
            # Check if synchronization is needed based on the index of the last block
            last_block_index = entries[-1]["index"]
            if last_block_index-1 > self.blockchain.get_last_block_index():
                print("must request sync")
                # Synchronize blockchain
                self.request_sync_blockchain(self.leader_id)
            else: 
                block = Block(
                    index=entries[-1]["index"],
                    timestamp=entries[-1]["timestamp"],
                    data=entries[-1]["data"],
                    previous_hash=entries[-1]["previous_hash"]
                )
                self.blockchain.add_block(block)

        # Send a response to the leader
        response = {
            "type": "append_entries_response",
            "success": True,  # You can modify this based on your implementation
            "term": self.current_term,
        }

        self.node.send_message(self.leader_id, response)

    def handle_client_request(self, message):
        print("in", self.state, self.leader_id)
        if self.state == "leader":
            # If this node is the leader, process the client request
            self.process_client_request(message)

    def handle_vote_response(self, message):
        vote_granted = message.get("vote_granted")
        term = message.get("term")

        if term > self.current_term:
            # If the responding node has a higher term, update the current term and become a follower
            self.current_term = term
            self.voted_for = None
            self.state = "follower"
            self.leader_id = None
            print(f"Node {self.node_id} updated to follower due to higher term {term}")

        elif vote_granted:
            # Increment the votes received if the vote is granted
            self.votes_received += 1
            self.leader_id = self.node_id
            print(f"Node {self.node_id} received a vote. Total votes: {self.votes_received}")

        # Check if the node has received a majority of votes
        if self.votes_received > self.total_nodes // 2:
            print(f"Node {self.node_id} has received a majority of votes. Becoming leader.")
            self.state = "leader"

    def handle_sync_request(self, message):
        requester_id = message.get("requester_id")
        self.sync_blockchain(requester_id)

    def handle_sync_blockchain(self, message):
        print("sync")
        received_blocks = message.get("blocks", [])

        # Get the index of the last block in the local blockchain
        last_local_index = self.blockchain.get_last_block_index()

        for block_data in received_blocks:
            block_index = block_data["index"]

            # Add the block only if it is missing in the local blockchain
            if block_index > last_local_index:
                block = Block(
                    index=block_index,
                    timestamp=block_data["timestamp"],
                    data=block_data["data"],
                    previous_hash=block_data["previous_hash"]
                )
                self.blockchain.add_block(block)

    def start_election(self):
        logging.info("Node %s is starting an election for term %s", self.node_id, self.current_term + 1)

        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = 1  # Vote for itself

        print("Requesting votes from other nodes")

        # Prepare the request_vote message
        request_vote_message = {
            "type": "request_vote",
            "candidate_id": self.node_id,
            "candidate_term": self.current_term,
        }

        # Ask for votes from other nodes
        self.node.broadcast_message(request_vote_message)

    def process_client_request(self, client_request):
        previous_blocks = self.blockchain.blocks

        block = Block(
            index= self.current_term,
            timestamp= "1111",
            data= client_request.get("content"),
            previous_hash= previous_blocks[-1].current_hash
        )
        self.blockchain.add_block(block)

        append_entries_message = {
            "type": "append_entries",
            "leader_id": self.node_id,
            "leader_term": self.current_term,
            "entries": [block.to_dict()],  # Include the new log entry
        }

        # Broadcast the append_entries_message to other nodes
        self.node.broadcast_message(append_entries_message)

    def request_vote_from_node(self, node_id):
        logging.info("Node %s is requesting a vote from Node %s for term %s", self.node_id, node_id, self.current_term)

        request_vote_message = {
            "type": "request_vote",
            "candidate_id": self.node_id,
            "candidate_term": self.current_term,
        }

        # Simulate network delay by sleeping for a short time
        time.sleep(random.uniform(0, 0.1))

        # Send the request for vote to the target node
        threading.Thread(target=self.send_message, args=(node_id, request_vote_message)).start()

    def request_sync_blockchain(self, target_node_id):
        # Get the index of the last block in the local blockchain
        last_block_index = self.blockchain.get_last_block_index()

        # Create a sync request message with the index of the last block
        sync_request_message = {
            "type": "sync_request",
            "requester_id": self.node_id,
            "last_block_index": last_block_index,
        }

        # Send the sync request message to the target node
        self.node.send_message(target_node_id, sync_request_message)

    def send_append_entries(self, node_id, entries):
        """
        Implement sending AppendEntries RPC to another node
        """
        pass

    def send_heartbeats(self):
        heartbeat_message = {
            "type": "append_entries",
            "leader_id": self.node_id,
            "leader_term": self.current_term,
            "entries": [],  # Heartbeats typically have empty entries
        }

        self.node.broadcast_message(heartbeat_message)

    def sync_blockchain(self, requesting_peer_id):
        blocks_to_send = self.blockchain.blocks
        sync_message = {
            "type": "sync_blockchain",
            "blocks": [block.to_dict() for block in blocks_to_send],
        }
        print("sending:", sync_message)
        self.node.send_message(requesting_peer_id, sync_message)

    def run(self):
        while True:
            current_time = time.time()

            if self.state == "follower" and current_time - self.last_heartbeat_time > self.election_timeout:
                # The election timeout has expired, start a new election
                self.state = "candidate"
                print(f"starting election with node: {self.node_id}")
                self.start_election()

            elif self.state == "candidate":
                # Run election
                self.start_election()
                time.sleep(random.uniform(1, 5))  # Simulate election timeout

            elif self.state == "leader":
                self.send_heartbeats()

            time.sleep(0.1)

    def reset_election_timeout(self):
        self.election_timeout = random.uniform(5, 15)
        self.last_heartbeat_time = time.time()
