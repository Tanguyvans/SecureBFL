settings = {
    "name_dataset": "cifar",  # "cifar" or "mnist" or "alzheimer"
    "arch": "simpleNet",  # "simpleNet" or "CNNCifar" or "CNNMnist" or "mobilenet"
    "pretrained": True,
    "patience": 2,
    "batch_size": 32,
    "n_epochs": 5,
    "numberOfNodes": 3,
    "numberOfClientsPerNode": 6,
    "min_number_of_clients_in_cluster": 3,
    "coef_usefull": 2,
    "poisonned_number": 0,
    "n_rounds": 20,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,
    "choice_scheduler": "StepLR",
    "step_size": 1,
    "gamma": 0.1,
    "diff_privacy": False,
    "secret_sharing": "additif",  # "additif" or "shamir"
    "k": 1,
    "m": 3
}