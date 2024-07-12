from going_modular.data_setup import load_data
from going_modular.utils import choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct
from going_modular.model import Net
from going_modular.engine import train, test


#%%
if __name__ == '__main__':
    # %% Parameters
    arch = 'simpleNet'  # 'CNNMnist'  # "CNNCifar"  # "mobilenet"
    name_dataset = 'cifar'  # "cifar"
    data_root = "data/"
    directory = 'models/'  # Update this path
    device = "mps"

    batch_size = 512
    n_epochs = 20
    lr = 0.001

    choice_loss = "cross_entropy"
    choice_scheduler = "StepLR"
    choice_optimizer = "Adam"

    # Set device
    device = choice_device(device)

    # %% read data with a private dataset (Energy)
    trainloader, valloader, testloader, classes = load_data(0,
                                                            1,
                                                            name_dataset=name_dataset,
                                                            data_root="./data",
                                                            resize=32 if arch == 'alzheimer' else None,
                                                            batch_size=batch_size)

    # %% Define the model
    model = Net(num_classes=len(classes), arch=arch).to(device)

    # %% Defining loss function and optimizer
    criterion = fct_loss(choice_loss)
    optimizer = choice_optimizer_fct(model, choice_optim=choice_optimizer, lr=lr, weight_decay=1e-6)
    scheduler = choice_scheduler_fct(optimizer, choice_scheduler=choice_scheduler, step_size=10, gamma=0.1)

    # %% Train the model
    results = train(0, model, trainloader, valloader, epochs=n_epochs,
                    loss_fn=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
                    dp=False)

    # %%Evaluate the model
    test_loss, test_acc, _, _, _ = test(model, testloader, criterion, device=device)

    # %% visualizations