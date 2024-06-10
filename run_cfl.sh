#!/bin/bash

# Check if the number of clients was passed as a parameter
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_clients>"
    exit 1
fi

num_clients=$1

# Activate the Python environment
source env/bin/activate

python3 cfl/server.py &
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