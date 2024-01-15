import time
import threading
import logging

from node import Node
from client import Client, data_preparation


if __name__ == "__main__": 
    logging.basicConfig(level=logging.DEBUG)

    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'

    numberOfClients = 3
    poisonned_number = 0
    epochs = 20

    with open("output.txt", "w") as f: 
        f.write("")

    train_sets = data_preparation(train_path, numberOfClients)
    test_sets = data_preparation(test_path, numberOfClients)

    for i in range(poisonned_number): 
        train_sets[i][1] = train_sets[i][1].replace({0: 1, 1: 0})
        test_sets[i][1] = test_sets[i][1].replace({0: 1, 1: 0})

    node1 = Node(id="n1", host="127.0.0.1", port=6010, consensus_protocol="pbft", batch_size=256, train=train_sets[0],
                 test=test_sets[0])
    node2 = Node(id="n2", host="127.0.0.1", port=6011, consensus_protocol="pbft", batch_size=256, train=train_sets[0],
                 test=test_sets[0])
    node3 = Node(id="n3", host="127.0.0.1", port=6012, consensus_protocol="pbft", batch_size=256, train=train_sets[0],
                 test=test_sets[0])

    client1 = Client(id="c1", host="127.0.0.1", port=5010, batch_size=256, train=train_sets[0], test=test_sets[0])
    client2 = Client(id="c2", host="127.0.0.1", port=5011, batch_size=256, train=train_sets[1], test=test_sets[1])
    client3 = Client(id="c3", host="127.0.0.1", port=5012, batch_size=256, train=train_sets[2], test=test_sets[2])

    client1.add_connections("c2", 5011)
    client1.add_connections("c3", 5012)

    client2.add_connections("c1", 5010)
    client2.add_connections("c3", 5012)

    client3.add_connections("c1", 5010)
    client3.add_connections("c2", 5011)

    client1.update_node_connection("n1", 6010)
    client2.update_node_connection("n1", 6010)
    client3.update_node_connection("n1", 6010)

    node1.add_peer(peer_id="n2", peer_address=("localhost", 6011))
    node1.add_peer(peer_id="n3", peer_address=("localhost", 6012))

    node2.add_peer(peer_id="n1", peer_address=("localhost", 6010))
    node2.add_peer(peer_id="n3", peer_address=("localhost", 6012))

    node3.add_peer(peer_id="n1", peer_address=("localhost", 6010))
    node3.add_peer(peer_id="n2", peer_address=("localhost", 6011))

    node1.add_client(client_id="c1", client_address=("localhost", 5010))
    node1.add_client(client_id="c2", client_address=("localhost", 5011))
    node1.add_client(client_id="c3", client_address=("localhost", 5012))

    node1.create_cluster([client1.id, client2.id, client3.id])

    threading.Thread(target=client1.start_server).start()
    threading.Thread(target=client2.start_server).start()
    threading.Thread(target=client3.start_server).start()

    threading.Thread(target=node1.start_server).start()
    threading.Thread(target=node2.start_server).start()
    threading.Thread(target=node3.start_server).start()

    node1.create_first_global_model_request()

    time.sleep(10)

    for i in range(5): 
        frag_weights_1 = client1.train()
        frag_weights_2 = client2.train()
        frag_weights_3 = client3.train()

        client1.send_frag_clients(frag_weights_1)
        client2.send_frag_clients(frag_weights_2)
        client3.send_frag_clients(frag_weights_3)

        client1.send_frag_node()
        client2.send_frag_node()
        client3.send_frag_node()

        time.sleep(5)

        if i > 1 and i % 2 == 0: 
            node1.create_global_model()

    time.sleep(5)

    node1.blockchain.print_blockchain()
    node2.blockchain.print_blockchain()
    node3.blockchain.print_blockchain()

    node1.blockchain.save_chain_in_file("node1.txt")
    node2.blockchain.save_chain_in_file("node2.txt")
    node3.blockchain.save_chain_in_file("node3.txt")
