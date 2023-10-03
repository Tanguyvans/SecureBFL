# proto

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. block.proto

## a faire

verifier la validiter du block avant de l'ajouter
garder la chaine la plus longue en cas de doublons
broadcast -> car pour l'instant reaseau entrièrmeent relié
