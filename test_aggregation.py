import copy
import numpy as np
from flwr.server.strategy.aggregate import aggregate

from going_modular.data_setup import (load_data_from_path, splitting_dataset, TensorDataset,
                                      train_test_split, DataLoader, torch)
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct)
from going_modular.model import Net
from going_modular.engine import train, test


# %%
def average_weights(models):
    # Obtenez les state_dict des modèles
    state_dicts = [model.state_dict() for model in models]

    # Créer un dictionnaire pour stocker les poids moyens
    avg_state_dict = copy.deepcopy(state_dicts[0])

    # Pour chaque clé (paramètre) dans le state_dict, calculer la moyenne des poids
    for key in avg_state_dict:
        avg_state_dict[key] = torch.stack([state_dict[key] for state_dict in state_dicts], dim=0).mean(dim=0)

    return avg_state_dict


# Extract the weights from the models and convert them to the format required for Flower
def get_weights(model):
    return [(param.data.clone().cpu().numpy()) for param in model.parameters()]


# Apply the aggregated weights to the model
def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def compare_weights(dico1, dico2, tol=1e-6):
    """
    Compare the weights of two PyTorch model weight dictionaries with a given tolerance
    :param dico1: state_dict of the first model
    :param dico2: state_dict of the second model
    :param tol: tolerance for the comparison of the weights (default: 1e-6)
    :return:
    """
    # compare the numpy arrays with a tolerance
    for key, val in dico1.items():
        if np.allclose(
                val.cpu().numpy(),
                dico2[key].cpu().numpy(),
                atol=tol
        ):
            print(f"The values of the numpy arrays of {key} are equal with a tolerance of {tol}")
        else:
            print(f"Difference for the numpy arrays of {key}:")
            print("We restart the comparison with a lighter tolerance:")

            compare_weights({key: dico1[key]}, {key: dico2[key]}, tol=tol*10)


def round_i(n_clients, model, trainloaders, valloaders, testloaders, criterion,
            choice_optimizer="Adam", choice_scheduler="StepLR", n_epochs=10,  lr_init=0.001, step_size=3, gamma=0.5,
            patience=2, device="gpu", tol=1e-6):
    # Try to train n_clients model and aggregate them to verify the influence on accuracy after aggregation
    # same weights for the init round for each client else replace by [Net(...) for _ in range(n_clients)]
    list_models = [copy.deepcopy(model) for _ in range(n_clients)]

    # Train each model
    for i in range(n_clients):
        # for each model
        print(f"Training model {i + 1}")
        lr_local = lr_init
        optimizer = choice_optimizer_fct(list_models[i], choice_optim=choice_optimizer, lr=lr_local, weight_decay=1e-6)
        scheduler = choice_scheduler_fct(optimizer, choice_scheduler=choice_scheduler, step_size=step_size, gamma=gamma)

        results = train(0, list_models[i], trainloaders[i], valloaders[i], epochs=n_epochs,
                        loss_fn=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
                        dp=False, patience=patience)

    # Evaluate each model
    for i in range(n_clients):
        print(f"Evaluating model {i + 1}")
        test_loss, test_acc, *_ = test(list_models[i], testloaders[i], criterion, device=device)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

    # Aggregate the models
    # Compute the average of the weights
    avg_state_dict = average_weights(list_models)

    # create a aggregated model and load the average weights
    model_aggregated = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    model_aggregated.load_state_dict(avg_state_dict)

    # check if the updated model is the same as the aggregated model
    print(torch.equal(avg_state_dict['model.fc3.weight'], model_aggregated.state_dict()['model.fc3.weight']))

    # redo the aggregation using the aggregate function from Flower
    weights = [get_weights(model) for model in list_models]

    # Add an arbitrary update weight for each model (here 10)
    weights_with_updates = [(weights[i], 10) for i in range(n_clients)]

    # Use the aggregate function from Flower to calculate the average weights
    parameters_aggregated = aggregate(weights_with_updates)

    # Create a new model for the aggregated weights
    model_aggregated2 = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    set_weights(model_aggregated2, parameters_aggregated)

    # Check that the aggregated weights are the same as the previous ones
    """
    There is a difference between the aggregated weights depending on the number of examples.
    Even if the same number of examples is used by each client,
    and therefore the weighted average becomes an arithmetic average,
    there is a difference in the rounded values (see below)
    """

    compare_weights(model_aggregated.state_dict(), model_aggregated2.state_dict(), tol=tol)

    return model_aggregated, model_aggregated2, list_models


# todo:
#  tester en initialisant avec des modèles différents pour voir si cela influence la convergence
#  tester avec smpc
#%%
if __name__ == '__main__':
    # %% Parameters
    arch = 'CNNCifar'  # "simpleNet"
    pretrained = True

    name_dataset = 'cifar'  # "cifar"  # "alzheimer"
    data_root = "data/"
    directory = 'models/'  # Update this path

    device = "mps"
    batch_size = 32
    n_epochs = 10  # 15
    patience = 3  # 5

    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = "StepLR"  # None
    lr = 0.001  # 1e-4  # 0.001
    step_size = 3
    gamma = 0.5

    n_clients = 5
    n_rounds = 5  # 20
    tol = 1e-6

    # Set device
    device = choice_device(device)

    criterion = fct_loss(choice_loss)

    # %% read data with a private dataset
    # Case for the classification problems

    dataset_train, dataset_test = load_data_from_path(resize=32 if name_dataset == 'alzheimer' else None,
                                                      name_dataset=name_dataset, data_root="./data")
    classes = dataset_train.classes

    train_sets = splitting_dataset(dataset_train, n_clients)
    test_sets = splitting_dataset(dataset_test, n_clients)
    train_loaders, val_loaders, test_loaders = [], [], []
    for i in range(n_clients):
        print(f"Number of samples for client {i + 1}: {len(train_sets[i][1])}")
        x_train, y_train = train_sets[i]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=None)
        train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
        val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))

        x_test, y_test = test_sets[i]
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        train_loaders.append(DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True))
        val_loaders.append(DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, drop_last=True))
        test_loaders.append(DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True))

    # Evaluate the aggregated models with the whole test data
    x_test, y_test = splitting_dataset(dataset_test, 1)[0]
    test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # %% start round
    model_aggregated2 = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    for i in range(n_rounds):
        print(f"\n\t /////////// Round {i + 1}\t ///////////")
        model_aggregated, model_aggregated2, list_models = round_i(n_clients, model_aggregated2,
                                                                   train_loaders, val_loaders, test_loaders,
                                                                   criterion, choice_optimizer, choice_scheduler,
                                                                   n_epochs, lr, step_size, gamma, patience, device, tol)

    # %% Evaluate the aggregated models with the test data of each client
    for i in range(n_clients):
        print(f"//////////\tEvaluating aggregated model with the test data of client {i + 1}\t//////////")

        # Evaluate the aggregated model with arithmetic average
        test_loss, test_acc, *_ = test(model_aggregated, test_loaders[i], criterion, device=device)
        print(f"Arithmetic average | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

        # Evaluate the aggregated model with the weighted average by Flower
        test_loss, test_acc, *_ = test(model_aggregated2, test_loaders[i], criterion, device=device)
        print(f"weighted average | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

        print(f"Evaluating aggregated models with the whole test data")
        # The test showsn a little difference (0.10% accuracy) even if the weights are same to the 1e-6 tolerance
        # So apply SMPC can to do a little difference too (because weights are not exactly the same)

    # %% Evaluate the models that have been aggregated by arithmetic average
    test_loss, test_acc, *_ = test(model_aggregated, test_loader, criterion, device=device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

    # Evaluate the models that have been aggregated by weighted average (from Flower function)
    test_loss, test_acc, *_ = test(model_aggregated2, test_loader, criterion, device=device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %\n")

    # best results :

    # Test loss: 0.9943 | Test accuracy: 72.29 %    (without scheduler but with 72.13 % and loss=1.7364)
    # Test loss: 0.9949 | Test accuracy: 72.27 %    (without scheduler but with 72.13 % and loss=1.7349)

    # %% Evaluate the local models with the whole test data
    for i in range(n_clients):
        print(f"\nEvaluating model {i + 1} with the whole test data")
        test_loss, test_acc, *_ = test(list_models[i], test_loader, criterion, device=device)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

    # best results:
    # Model 1  Test loss: 2.1812 | Test accuracy: 69.44 %
    # Model 2  Test loss: 2.1977 | Test accuracy: 68.78 %
    # Model 3  Test loss: 2.7116 | Test accuracy: 69.53 %
    # Model 4  Test loss: 2.1531 | Test accuracy: 69.25 %
    # Model 5  Test loss: 2.5886 | Test accuracy: 69.12 %
    # aggregated model  better after more rounds but before each local model is better.
    # For example, after 1 round, accuracy of the aggregated model is 10% while the accuracy of the local models is 35%

    # without scheduler, I obtained less good accuracy (66%)
    # with a better loss (1.5) for each models BUT a better aggregate model (loss and acc)