import copy
import os
import numpy as np
import json
from flwr.server.strategy.aggregate import aggregate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from going_modular.data_setup import (load_data_from_path, splitting_dataset, TensorDataset,
                                      train_test_split, DataLoader, torch)
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct,
                                 get_parameters, set_parameters)
from going_modular.model import Net
from going_modular.engine import train, test


# %%
def average_parameters(models):
    """
    Compute the average of the parameters of the models
    :param models:
    :return:
    """
    # Obtenez les state_dict des modèles
    state_dicts = [model.state_dict() for model in models]

    # Créer un dictionnaire pour stocker les poids moyens
    avg_state_dict = copy.deepcopy(state_dicts[0])

    # Pour chaque clé (paramètre) dans le state_dict, calculer la moyenne des poids
    for key in avg_state_dict:
        #  Vérifiez si le paramètre est optimisable (poids ou biais)
        # if 'weight'  in key or 'bias' in key
        avg_state_dict[key] = torch.stack([state_dict[key] for state_dict in state_dicts], dim=0).float().mean(dim=0)

    return avg_state_dict


# Extract the weights from the models and convert them to the format required for Flower
def set_parameters2(model, parameters):
    # Charger les paramètres optimisables (poids et biais) dans le modèle
    param_dict = zip(model.parameters(), parameters)

    for param, new_val in param_dict:
        param.data = torch.tensor(new_val, dtype=param.data.dtype).to(param.device)


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
        if "weight" in key or "bias" in key:
            if np.allclose(
                    val.cpu().numpy(),
                    dico2[key].cpu().numpy(),
                    atol=tol
            ):
                print(f"The values of the numpy arrays of {key} are equal with a tolerance of {tol}")
            else:
                print(f"Difference for the numpy arrays of {key}:")
                print("We restart the comparison with a lighter tolerance:")

                compare_weights({key: dico1[key]}, {key: dico2[key]}, tol * 10)


def round_i(n_clients, model, trainloaders, valloaders, testloaders, criterion,
            choice_optimizer="Adam", choice_scheduler="StepLR", epochs=10,  lr_init=0.001, step_size=3, gamma=0.5,
            patience=2, device="gpu"):
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

        os.makedirs("models/scratch/", exist_ok=True)
        _ = train(0, list_models[i], trainloaders[i], valloaders[i], epochs=epochs,
                  loss_fn=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
                  dp=False, patience=patience, save_model=f"models/scratch/best_model_scratch_{i}.pth")

    # Evaluate each model
    for i in range(n_clients):
        print(f"Evaluating model {i + 1}")
        test_loss, test_acc, *_ = test(list_models[i], testloaders[i], criterion, device=device)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

    # Aggregate the models
    # Compute the average of the parameters (weights, bias, ...)
    avg_state_dict = average_parameters(list_models)

    # create an aggregated model and load the average parameters
    model_aggregated = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    model_aggregated.load_state_dict(avg_state_dict, strict=False)
    keyy = list(avg_state_dict.keys())[-2]

    # check if the updated model is the same as the aggregated model
    print(torch.equal(avg_state_dict[keyy], model_aggregated.state_dict()[keyy]))

    # redo the aggregation using the aggregate function from Flower
    list_parameters = [get_parameters(model) for model in list_models]

    # Add an arbitrary update weight for each model (here 10)
    weights_with_updates = [(list_parameters[i], 10) for i in range(n_clients)]

    # Use the aggregate function from Flower to calculate the average parameters
    parameters_aggregated = aggregate(weights_with_updates)

    # Create a new model for the aggregated weights
    model_aggregated2 = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    set_parameters(model_aggregated2, parameters_aggregated)

    return model_aggregated, model_aggregated2, list_models


# todo:
#  tester avec smpc
#%%
if __name__ == '__main__':
    # %% Parameters
    arch = 'mobilenet'  # "simpleNet"
    pretrained = True

    name_dataset = 'cifar'  # "cifar"  # "alzheimer"
    data_root = "data/"

    device = "mps"
    length = 32
    batch_size = 32
    n_epochs = 5  # 15
    patience = 2  # 5

    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = "StepLR"  # None
    lr = 0.001  # 1e-4  # 0.001
    step_size = 3
    gamma = 0.5

    n_clients = 6
    n_rounds = 5  # 20
    tolerance = 1e-6

    json_dict = {
        'settings': {
            'arch': arch,
            'pretrained': pretrained,
            'name_dataset': name_dataset,
            'device': device,
            'patience': patience,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'choice_loss': choice_loss,
            'choice_optimizer': choice_optimizer,
            'lr': lr,
            'choice_scheduler': choice_scheduler,
            'step_size': step_size,
            'gamma': gamma,
            "n_clients": n_clients,
            "n_rounds": n_rounds,
            "length": length,
            "tolerance": tolerance,
        }
    }
    with open("results/scratch/" + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    # Set device
    device = choice_device(device)

    criterion = fct_loss(choice_loss)

    # %% read data with a private dataset
    # Case for the classification problems

    dataset_train, dataset_test = load_data_from_path(resize=length,  # resize=32 if name_dataset == 'alzheimer' else None,
                                                      name_dataset=name_dataset, data_root="./data")
    classes = dataset_train.classes

    train_sets = splitting_dataset(dataset_train, n_clients)
    test_sets = splitting_dataset(dataset_test, n_clients)
    train_loaders, val_loaders, test_loaders = [], [], []
    for client_id in range(n_clients):
        print(f"Number of samples for client {client_id + 1}: {len(train_sets[client_id][1])}")
        x_train, y_train = train_sets[client_id]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=None)
        train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
        val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))

        x_test, y_test = test_sets[client_id]
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        train_loaders.append(DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True))
        val_loaders.append(DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, drop_last=True))
        test_loaders.append(DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True))

    # Evaluate the aggregated models with the whole test data
    x_test, y_test = splitting_dataset(dataset_test, 1)[0]
    test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # %% start round
    model_agg2 = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)
    models_list = []
    model_agg = None
    for id_round in range(n_rounds):
        print(f"\n\t /////////// Round {id_round + 1}\t ///////////")
        model_agg, model_agg2, models_list = round_i(n_clients, model_agg2,
                                                     train_loaders, val_loaders, test_loaders,
                                                     criterion, choice_optimizer, choice_scheduler,
                                                     n_epochs, lr, step_size, gamma, patience, device)

        # Test the aggregated models with the whole test data
        loss, acc, *_ = test(model_agg2, test_loader, criterion, device=device)
        print(f"After Round {id_round + 1 }| Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %")

    # %% Check that the aggregated weights are the same as the previous ones
    """
    There is a difference between the aggregated weights depending on the number of examples.
    Even if the same number of examples is used by each client,
    and therefore the weighted average becomes an arithmetic average,
    there is a difference in the rounded values (see below)
    """

    compare_weights(model_agg.state_dict(), model_agg2.state_dict(), tol=tolerance)

    # %% Evaluate the aggregated models with the test data of each client
    for id_client in range(n_clients):
        print(f"//////////\tEvaluating aggregated model with the test data of client {id_client + 1}\t//////////")

        # Evaluate the aggregated model with arithmetic average
        loss, acc, *_ = test(model_agg, test_loaders[id_client], criterion, device=device)
        print(f"Arithmetic average | Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %")

        # Evaluate the aggregated model with the weighted average by Flower
        loss, acc, *_ = test(model_agg2, test_loaders[id_client], criterion, device=device)
        print(f"weighted average | Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %")

        print(f"Evaluating aggregated models with the whole test data")
        # The test shows a little difference (0.37% accuracy) even if the weights are same to the 1e-6 tolerance
        # So apply SMPC can to do a little difference too (because weights are not exactly the same)

    # %% Evaluate the models that have been aggregated by arithmetic average
    loss, acc, *_ = test(model_agg, test_loader, criterion, device=device)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %")

    # Evaluate the models that have been aggregated by weighted average (from Flower function)
    loss, acc, *_ = test(model_agg2, test_loader, criterion, device=device)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %\n")

    # best results (5 clients):
    # Test loss: 0.9943 | Test accuracy: 72.29 %    (without scheduler but with 72.13 % and loss=1.7364)
    # Test loss: 0.9949 | Test accuracy: 72.27 %    (without scheduler but with 72.13 % and loss=1.7349)

    # best results (18 clients):
    # Test loss: 1.0842 | Test accuracy: 63.56 %
    # Test loss: 1.0840 | Test accuracy: 63.54 %

    # if each client initializes with a different model, the accuracy of the aggregated model is at 54.7%.

    # news:
    # test avec mobilenet corrigé et l'agrégation d'uniquement les paramètres optimisables:
    # Test loss: 0.8828 | Test accuracy: 72.81 %
    # Test loss: 30.1978 | Test accuracy: 14.10 %   # Pourquoi l'agrégation de flower détruit les résultats ?

    # Test avec mobileNet corrigé et l'agrégation pour 5 clients:
    # Test loss: 1.4750 | Test accuracy: 61.71 %  # uniquement les paramètres optimisables
    # Test loss: 1.6717 | Test accuracy: 72.73 %  # agrégation de tous les paramètres

    # Test avec mobileNet corrigé et l'agrégation de tous les paramètres pour 5 clients:
    # Test loss: 1.6689 | Test accuracy: 70.80 %
    # Test loss: 1.6694 | Test accuracy: 70.78 %

    # %% Evaluate the local models with the whole test data
    for id_client in range(n_clients):
        print(f"\nEvaluating model {id_client + 1} with the whole test data")
        loss, acc, *_ = test(models_list[id_client], test_loader, criterion, device=device)
        print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.2f} %")

    # best results (5 clients):
    # Model 1  Test loss: 2.1812 | Test accuracy: 69.44 %
    # Model 2  Test loss: 2.1977 | Test accuracy: 68.78 %
    # Model 3  Test loss: 2.7116 | Test accuracy: 69.53 %
    # Model 4  Test loss: 2.1531 | Test accuracy: 69.25 %
    # Model 5  Test loss: 2.5886 | Test accuracy: 69.12 %
    # aggregated model  better after more rounds but before each local model is better.
    # For example, after 1 round, accuracy of the aggregated model is 10% while the accuracy of the local models is 35%

    # without scheduler, I obtained less good accuracy (66%)
    # with a better loss (1.5) for each model BUT a better aggregate model (loss and acc)

    # best results (18 clients):
    # Model 1 Test loss: 1.3220 | Test accuracy: 57.49 %
    # Model 2 Test loss: 1.3540 | Test accuracy: 58.67 %
    # Model 3 Test loss: 1.5739 | Test accuracy: 58.50 %
    # Model 4 Test loss: 1.3182 | Test accuracy: 58.49 %
    # Model 5 Test loss: 1.3110 | Test accuracy: 58.22 %
    # Model 6 Test loss: 1.4334 | Test accuracy: 57.62 %
    # Model 7 Test loss: 1.3367 | Test accuracy: 58.96 %
    # Model 8 Test loss: 1.3400 | Test accuracy: 58.50 %
    # Model 9 Test loss: 1.2919 | Test accuracy: 58.69 %
    # Model 10 Test loss: 1.3899 | Test accuracy: 58.15 %
    # Model 11 Test loss: 1.3450 | Test accuracy: 58.53 %
    # Model 12 Test loss: 1.5000 | Test accuracy: 57.65 %
    # Model 13 Test loss: 1.4887 | Test accuracy: 56.84 %
    # Model 14 Test loss: 1.4560 | Test accuracy: 57.00 %
    # Model 15 Test loss: 1.3693 | Test accuracy: 58.50 %
    # Model 16 Test loss: 1.3414 | Test accuracy: 58.37 %
    # Model 17 Test loss: 1.3840 | Test accuracy: 57.70 %
    # Model 18 Test loss: 1.3074 | Test accuracy: 58.38 %

    # If each client initializes with a different model, the accuracy of the aggregated model is at 52.4% max.
