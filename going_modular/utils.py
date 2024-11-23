import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
# import scikitplot as skplt
import os
import pandas as pd
import seaborn as sn
import threading
from collections import OrderedDict
from typing import List


def initialize_parameters(settings, training_approach):
    settings["data_root"] = "Data"
    settings["roc_path"] = None  # "roc"
    settings["matrix_path"] = None  # "matrix"
    settings["save_results"] = f"results/{training_approach}/"
    settings["save_model"] = f"models/{training_approach}/"

    # clients
    training_barrier = threading.Barrier(settings['number_of_clients_per_node'])

    if training_approach.lower() == "cfl":
        settings["n_clients"] = settings["number_of_clients_per_node"] * settings["number_of_nodes"]
        [settings.pop(key, None) for key in ["number_of_clients_per_node",
                                             "number_of_nodes",
                                             "min_number_of_clients_in_cluster",
                                             "k",
                                             "m",
                                             "secret_sharing"]]

    elif training_approach.lower() == "bfl":
        if settings["m"] < settings["k"]:
            raise ValueError(
                "the number of parts used to reconstruct the secret must be greater than the threshold (k)")
        print("Number of Nodes: ", settings["number_of_nodes"],
              "\tNumber of Clients per Node: ", settings["number_of_clients_per_node"],
              "\tNumber of Clients per Cluster: ", settings["min_number_of_clients_in_cluster"], "\n")

    os.makedirs(settings["save_results"], exist_ok=True)
    return training_barrier, None

def sMAPE(outputs, targets):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) for evaluating the model.
    It is the sum of the absolute difference between the predicted and actual values divided by the average of
    the predicted and actual value, therefore giving a percentage measuring the amount of error :
    100/n * sum(|F_t - A_t| / (|F_t| + |A_t|) / 2) with t = 1 to n

    :param outputs : predicted values
    :param targets: real values
    :return: sMAPE
    """

    return 100 / len(targets) * np.sum(np.abs(outputs - targets) / (np.abs(outputs + targets)) / 2)

def fct_loss(choice_loss):
    """
    A function to choose the loss function

    :param choice_loss: the choice of the loss function
    :return: the loss function
    """
    choice_loss = choice_loss.lower()
    print("choice_loss : ", choice_loss)

    if choice_loss == "cross_entropy":
        loss = torch.nn.CrossEntropyLoss()

    elif choice_loss == "binary_cross_entropy":
        loss = torch.nn.BCELoss()

    elif choice_loss == "bce_with_logits":
        loss = torch.nn.BCEWithLogitsLoss()

    elif choice_loss == "smape":
        loss = sMAPE

    elif choice_loss == "mse":
        loss = torch.nn.MSELoss(reduction='mean')

    elif choice_loss == "scratch":
        return None

    else:
        print("Warning problem : unspecified loss function")
        return None

    print("loss : ", loss)

    return loss

def choice_optimizer_fct(model, choice_optim="Adam", lr=0.001, momentum=0.9, weight_decay=1e-6):
    """
    A function to choose the optimizer

    :param model: the model
    :param choice_optim: the choice of the optimizer
    :param lr: the learning rate
    :param momentum: the momentum
    :param weight_decay: the weight decay
    :return: the optimizer
    """

    choice_optim = choice_optim.lower()

    if choice_optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    elif choice_optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                     amsgrad='adam' == 'amsgrad', )

    elif choice_optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        print("Warning problem : unspecified optimizer")
        return None

    return optimizer

def choice_scheduler_fct(optimizer, choice_scheduler=None, step_size=30, gamma=0.1, base_lr=0.0001, max_lr=0.1):
    """
    A function to choose the scheduler

    :param optimizer: the optimizer
    :param choice_scheduler: the choice of the scheduler
    :param step_size: the step size
    :param gamma: the gamma
    :param base_lr: the base learning rate
    :param max_lr: the maximum learning rate
    :return: the scheduler
    """

    if choice_scheduler:
        choice_scheduler = choice_scheduler.lower()

    if choice_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif choice_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif choice_scheduler == "cycliclr":
        # use per batch (not per epoch)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr)

    elif choice_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=0)

    elif choice_scheduler == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=gamma)

    elif choice_scheduler == "plateaulr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08, verbose=False)

    elif choice_scheduler is None:
        return choice_scheduler

    else:
        print("Warning problem : unspecified scheduler")
        # There are other schedulers like OneCycleLR, etc.
        # but generally, they are used per batch and not per epoch.
        # For example, OneCycleLR : total_steps = n_epochs * steps_per_epoch
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
        return None

    print("scheduler : ", scheduler)
    return scheduler

def choice_device(device):
    """
        A function to choose the device

        :param device: the device to choose (cpu, gpu or mps)
        """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    print("The device is : ", device)
    return device

def save_matrix(y_true, y_pred, path, classes):
    """
    Save the confusion matrix in the path given in argument.

    :param y_true: true labels (real labels)
    :param y_pred: predicted labels (labels predicted by the model)
    :param path: path to save the confusion matrix
    :param classes: list of the classes
    """
    # To get the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # To normalize the confusion matrix
    cf_matrix_normalized = cf_matrix / np.sum(cf_matrix) * 10

    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(cf_matrix_round, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    # skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)

    plt.savefig(path)
    plt.close()

def save_roc(targets, y_proba, path, nbr_classes):
    """
    Save the roc curve in the path given in argument.

    :param targets: true labels (real labels)
    :param y_proba: predicted labels (labels predicted by the model)
    :param path: path to save the roc curve
    :param nbr_classes: number of classes
    """

    y_true = np.zeros(shape=(len(targets), nbr_classes))  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='Worst case')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    # skplt.metrics.plot_roc(targets, y_proba)

    plt.savefig(path)
    plt.close()

def save_graphs(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"],
        title="Loss curves",
        path=path_save + "Loss_curves" + end_file)
    print("graphs saved in ", path_save)

def plot_graph(list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None):
    """
    Plot the graph of the list of points (list_xplot, list_yplot)
    :param list_xplot: list of list of points to plot (one line per curve)
    :param list_yplot: list of list of points to plot (one line per curve)
    :param x_label: label of the x-axis
    :param y_label: label of the y-axis
    :param curve_labels: list of labels of the curves (curve names)
    :param title: title of the graph
    :param path: path to save the graph
    """
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)

def get_parameters(model):
    # if we want to return only the optimized parameters
    # return [val.detach().cpu().numpy() for name, val in model.named_parameters() if 'weight'  in name or 'bias' in name]

    # excluding parameters of BN layers when using FedBN
    return [val.cpu().numpy() for name, val in model.state_dict().items() if 'bn' not in name]

def set_parameters(model, parameters: List[np.ndarray]) -> None:
    # Set model parameters from a list of NumPy ndarrays
    keys = [k for k in model.state_dict().keys() if 'bn' not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
