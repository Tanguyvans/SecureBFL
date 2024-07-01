#!/bin/bash

# Check if the number of clients and number of epochs were passed as parameters
#if [ "$#" -ne 3 ]; then
#    echo "Usage: $0 <num_clients> <rounds> <max_epochs>"
#    exit 1

#fi

num_clients=18
rounds=10
max_epochs=20
batch_size=32
dataset='cifar' # 'alzheimer', 'mnist', 'cifar'
arch='mobilenet' # 'mobilenet', 'resnet', 'simplenet'
device='mps' # 'cpu', 'mps'
data_path='./data/'
save_results='./results/FL/'
matrix_path='confusion_matrix'
roc_path='roc'
# Activate the Python environment
source env/bin/activate

# Start the server with the specified number of epochs
echo "Starting server with $rounds rounds and $max_epochs epochs"
python3 cfl/server.py --num_clients $num_clients --rounds $rounds --max_epochs $max_epochs --batch_size $batch_size --dataset $dataset --arch $arch --device $device &
server_pid=$!

# Start multiple clients in the background
for (( i=0; i<num_clients; i++ ))
do
    echo "Starting client $i"
    path_matrix="${matrix_path}_client_${i}.png"
    path_roc="${roc_path}_client_${i}.png"
    python3 cfl/client.py --partition_id $i --num_clients $num_clients  --batch_size $batch_size --dataset $dataset --arch $arch --max_epochs $max_epochs --device $device --data_path $data_path  --save_results $save_results --matrix_path $path_matrix --roc_path $path_roc &
    declare pid_$i=$!
done

# Wait for all clients to finish
for (( i=0; i<num_clients; i++ ))
do
    wait ${!pid_$i}
done

# Deactivate the Python environment
deactivate