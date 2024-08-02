settings = {
    "name_dataset": "cifar",  # "cifar" or "mnist" or "alzheimer"
    "arch": "mobilenet",  # "simpleNet" or "CNNCifar" or "CNNMnist" or "mobilenet"
    "pretrained": True,
    "patience": 3,
    "batch_size": 32,
    "n_epochs": 5,
    "number_of_nodes": 3,
    "number_of_clients_per_node": 6,
    "min_number_of_clients_in_cluster": 3,
    "coef_usefull": 2,   # 1.05
    "poisoned_number": 0,
    "n_rounds": 20,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,
    "choice_scheduler": "StepLR",  # "StepLR" or None
    "step_size": 5,
    "gamma": 0.5,
    "diff_privacy": False,
    "secret_sharing": "additif",  # "additif" or "shamir"
    "k": 3,
    "m": 3,
    "ts": 10
}
