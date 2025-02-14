settings = {
    "name_dataset": "cifar10",  # "cifar10" or "cifar100" or "caltech256"
    "arch": "mobilenet",  # "mobilenet" or "resnet18" or "shufflenet"
    "pretrained": True,
    "patience": 3,
    "batch_size": 32,
    "n_epochs": 15,
    "number_of_nodes": 3,
    "number_of_clients_per_node": 15,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": True,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

    "poisoned_number": 0,
    "n_rounds": 50,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,
    "choice_scheduler": "StepLR",  # "StepLR" or None
    "step_size": 3,
    "gamma": 0.5,
    "diff_privacy": False,
    "secret_sharing": "additif",  # "additif" or "shamir"
    "k": 3,
    "m": 3,
    "ts": 20
}
