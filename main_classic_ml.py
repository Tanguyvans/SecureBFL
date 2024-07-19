from going_modular.data_setup import (load_data_from_path, splitting_dataset, TensorDataset,
                                      train_test_split, DataLoader, torch)
from going_modular.utils import choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct
from going_modular.model import Net
from going_modular.engine import train, test

import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context


#%%
if __name__ == '__main__':
    # %% Parameters
    arch = 'SqueezeNet'  # "simpleNet" # "CNNCifar" # "mobilenet" # "SqueezeNet" # 'efficientnetB0'  # ''Resnet50'
    pretrained = True
    name_dataset = 'cifar'  # "cifar"
    data_root = "data/"
    directory = 'models/'  # Update this path
    device = "mps"
    patience = 2

    batch_size = 32
    n_epochs = 100
    lr = 0.001  # 1e-4  # 0.001

    choice_loss = "cross_entropy"
    choice_scheduler = "StepLR"  # 'cycliclr' #"StepLR"
    choice_optimizer = "Adam"

    # Set device
    device = choice_device(device)

    # %% read data with a private dataset
    # Case for the classification problems

    dataset_train, dataset_test = load_data_from_path(resize=32 if name_dataset == 'alzheimer' else None,
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
    scheduler = choice_scheduler_fct(optimizer, choice_scheduler=choice_scheduler, step_size=10, gamma=0.1)

    # %% Train the model
    results = train(0, model, trainloader, valloader, epochs=n_epochs,
                    loss_fn=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
                    dp=False, patience=patience)

    # %%Evaluate the model
    test_loss, test_acc, _, _, _ = test(model, testloader, criterion, device=device)

    # %% visualizations
