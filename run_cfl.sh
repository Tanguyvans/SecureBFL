#!/bin/bash

# Check if the number of clients and number of epochs were passed as parameters
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_clients> <number_of_epochs>"
    exit 1
fi

num_clients=$1
num_epochs=$2

# Activate the Python environment
source env/bin/activate

# Start the server with the specified number of epochs
echo "Starting server with $num_epochs epochs"
python3 cfl/server.py --epochs $num_epochs &
server_pid=$!

# Start multiple clients in the background
for (( i=0; i<num_clients; i++ ))
do
    echo "Starting client $i"
    python3 cfl/client.py --partition-id $i --num-clients $num_clients &
    declare pid_$i=$!
done

# Wait for all clients to finish
for (( i=0; i<num_clients; i++ ))
do
    wait ${!pid_$i}
done

# Deactivate the Python environment
deactivate