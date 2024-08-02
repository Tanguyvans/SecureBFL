import os
import json
from going_modular.data_setup import (load_data_from_path, splitting_dataset, TensorDataset,
                                      train_test_split, DataLoader, torch)
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct, save_matrix,
                                 save_roc, save_graphs)
from going_modular.model import Net
from going_modular.engine import train, test

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


#%%
if __name__ == '__main__':
    # %% Parameters
    arch = 'simpleNet'  # "simpleNet" # "CNNCifar" # "mobilenet" # "SqueezeNet" # 'efficientnetB0'  # ''Resnet50'
    pretrained = True
    name_dataset = 'alzheimer'  # "cifar"  # "alzheimer"
    data_root = "data/"
    save_results = "results/classic/"
    matrix_path = "matrix"
    roc_path = "roc"
    dir_model = "models/"
    name_model = f"{name_dataset}_ML_best_model.pth"
    device = "mps"
    patience = 5

    batch_size = 32
    n_epochs = 50

    choice_loss = "cross_entropy"
    choice_optimizer = "Adam"
    choice_scheduler = "StepLR"  # 'cycliclr' #"StepLR"
    lr = 0.001  # 1e-4  # 0.001
    step_size = 5
    gamma = 0.1
    length = 32  #  if name_dataset == 'alzheimer' else None
    # Set device
    device = choice_device(device)

    if dir_model:
        # Create the directory et les sous-dossiers if they don't exist
        os.makedirs(dir_model, exist_ok=True)

    # %% read data with a private dataset
    # Case for the classification problems

    dataset_train, dataset_test = load_data_from_path(resize=length,
                                                      name_dataset=name_dataset, data_root="./data")

    train_sets = splitting_dataset(dataset_train, 1)
    test_sets = splitting_dataset(dataset_test, 1)

    x_train, y_train = train_sets[0]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=None)
    train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
    val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))

    x_test, y_test = test_sets[0]
    test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_data, batch_size=batch_size)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size)

    classes = dataset_train.classes

    # %% Define the model
    model = Net(num_classes=len(classes), arch=arch, pretrained=pretrained).to(device)

    # %% Defining loss function and optimizer
    criterion = fct_loss(choice_loss)
    optimizer = choice_optimizer_fct(model, choice_optim=choice_optimizer, lr=lr, weight_decay=1e-6)
    scheduler = choice_scheduler_fct(optimizer, choice_scheduler=choice_scheduler, step_size=step_size, gamma=gamma)

    # %% Train the model
    results = train(0, model, trainloader, valloader, epochs=n_epochs,
                    loss_fn=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
                    dp=False, patience=patience, save_model=dir_model + name_model)

    save_graphs(save_results,
                results["stopping_n_epoch"] if results["stopping_n_epoch"] else n_epochs,
                results)

    # %%Evaluate the model
    test_loss, test_acc, y_pred, y_true, y_proba = test(model, testloader, criterion, device=device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f} %")

    # %% visualizations
    if save_results:
        os.makedirs(save_results, exist_ok=True)

        if matrix_path:
            save_matrix(y_true, y_pred,
                        save_results + matrix_path,
                        classes)

        if roc_path:
            save_roc(y_true, y_proba, save_results + roc_path,
                     # + f"_client_{self.cid}.png",
                     len(classes))

    # %% save results in json file
    json_dict = {
        "training_results": {**results, },
        "test_results": {'test_loss': test_loss, 'test_acc': test_acc},
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
            'length': length
        }
    }
    with open(save_results + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
