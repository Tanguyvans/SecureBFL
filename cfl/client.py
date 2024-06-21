import argparse
import warnings
from collections import OrderedDict
from typing import List
import numpy as np
import sys
import os

import flwr as fl
import torch


from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from going_modular.data_setup import splitting_dataset, NORMALIZE_DICT

from flowerclient import choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct, save_matrix, save_roc, save_graphs, PrivacyEngine, BatchMemoryManager, Net, TensorDataset, DataLoader


warnings.filterwarnings("ignore", category=UserWarning)


def train_step(model, dataloader, loss_fn, optimizer, device, scheduler):
    # Put model in training mode
    model.train()

    total = 0
    correct = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)

        # Calculate loss
        loss = loss_fn(y_pred, y_batch)

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        # Print the current loss and accuracy
        if total > 0:
            accuracy = 100 * correct / total
            # print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    if scheduler:
        scheduler.step()
    else:
        print("No scheduler")

    return loss.item(), accuracy


def train(model, dataloader, epochs, loss_fn, optimizer, scheduler, device="cpu",
          dp=False, delta=1e-5, max_physical_batch_size=256, privacy_engine=None):
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": []}  # , "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        if dp:
            with BatchMemoryManager(data_loader=dataloader,
                                    max_physical_batch_size=max_physical_batch_size,
                                    optimizer=optimizer) as memory_safe_data_loader:
                epoch_loss, epoch_acc = train_step(model, memory_safe_data_loader, loss_fn, optimizer, device, scheduler)
                epsilon = privacy_engine.get_epsilon(delta)
                tmp = f"(ε = {epsilon:.2f}, δ = {delta})"
        else:
            epoch_loss, epoch_acc = train_step(model, dataloader, loss_fn, optimizer, device, scheduler)
            tmp = ""

            # Print out what's happening
            print(
                f"\tTrain Epoch: {epoch + 1} \t"
                f"Train_loss: {epoch_loss:.4f} | "
                f"Train_acc: {epoch_acc:.4f} % | "
                # f"Validation_loss: {val_loss:.4f} | "
                # f"Validation_acc: {val_acc:.4f} %" + tmp
            )

            # Update results dictionary
            results["train_loss"].append(epoch_loss)
            results["train_acc"].append(epoch_acc)
            # results["val_loss"].append(val_loss)
            # results["val_acc"].append(val_acc)

    return results


def test(model, testloader, loss_fn, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total = 0
    correct = 0
    y_pred = []
    y_true = []
    y_proba = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 1. Forward pass
            output = model(x_batch)

            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            # 4. Calculate and accumulate accuracy
            _, predicted = torch.max(output, 1)  # np.argmax(output.detach().cpu().numpy(), axis=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            y_true.extend(y_batch.detach().cpu().numpy().tolist())  # Save Truth

            y_pred.extend(predicted.detach().cpu().numpy())  # Save Prediction

    model.train()
    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy, y_pred, y_true, np.array(y_proba)


def load_data(partition_id, num_clients, name_dataset="cifar", data_root="./data", resize=None, batch_size=256):
    data_folder = f"{data_root}/{name_dataset}"
    # Case for the classification problems
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[name_dataset])]
    if resize is not None:
        list_transforms = [transforms.Resize((resize, resize))] + list_transforms
    transform = transforms.Compose(list_transforms)

    if name_dataset == "cifar":
        dataset_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)

    elif name_dataset == "mnist":
        dataset_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

    elif name_dataset == "alzheimer":
        dataset_train = datasets.ImageFolder(data_folder + "/train", transform=transform)
        dataset_test = datasets.ImageFolder(data_folder + "/test", transform=transform)

    else:
        raise ValueError("The dataset name is not correct")

    train_sets = splitting_dataset(dataset_train, num_clients)
    test_sets = splitting_dataset(dataset_test, num_clients)

    x_train, y_train = train_sets[partition_id]
    # train_data = TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_train]), torch.tensor(y_train))
    train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))

    x_test, y_test = test_sets[partition_id]
    # TensorDataset(torch.stack([torchvision.transforms.functional.to_tensor(img) for img in x_test]), torch.tensor(y_test))
    test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    return train_loader, test_loader, dataset_train.classes


def get_parameters2(net) -> List[np.ndarray]:
    """
    Get the parameters of the network.
    :param net: Network to get the parameters (weights and biases)
    :return: list of parameters (weights and biases) of the network
    """

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, classes, trainloader, testloader, batch_size, epochs=1, model_choice="simplenet",
                 dp=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2,
                 device="gpu", lr=0.001, choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None,
                 save_results=None, matrix_path=None, roc_path=None):
        self.cid = cid
        self.classes = classes
        self.device = choice_device(device)
        self.model = Net(num_classes=len(self.classes), arch=model_choice).to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.epochs = epochs
        self.batch_size = batch_size

        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path

        self.criterion = fct_loss(choice_loss)
        self.optimizer = choice_optimizer_fct(self.model, choice_optim=choice_optimizer, lr=lr, weight_decay=1e-6)
        self.scheduler = choice_scheduler_fct(self.optimizer, choice_scheduler=choice_scheduler, step_size=10, gamma=0.1)

        self.dp = dp
        self.delta = delta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    def get_parameters(self, config):
        return get_parameters2(self.model)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        - Update the parameters of the local model with the parameters received from the server
        - Train the updated model on the local train dataset (x_train/y_train)
        - Return the updated model parameters and the number of examples used for training to the server
        (for 1 given round)

        args:
            parameters: parameters (list)
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        # Read values from config
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        # Use values provided by the config
        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')

        # Update local model parameters
        self.set_parameters(parameters)

        if self.dp:
            # With differential privacy
            print("We going to apply the differential privacy : ")
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                epochs=local_epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )

        results = train(self.model, self.trainloader, epochs=self.epochs,
                        loss_fn=self.criterion, optimizer=self.optimizer, scheduler=self.scheduler, device=self.device,
                        dp=self.dp, delta=self.delta, max_physical_batch_size=int(self.batch_size / 4),
                        privacy_engine=self.privacy_engine)

        #results = train(self.model, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=self.criterion,
        #                epochs=local_epochs, device=self.device,
        #                diff_privacy=self.dp, delta=PRIVACY_PARAMS['target_delta'],
        #                max_physical_batch_size=int(self.batch_size / 4), privacy_engine=self.privacy_engine
        #                ,problem_type=self.problem_type,
        #                scheduler=scheduler, early_stopping=self.early_stopping, patience=self.patience
        #                )

        # Save results
        if self.save_results:
            save_graphs(self.save_results, local_epochs, results, f"_Client {self.cid}")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}  # Use len(x_train) or len(y_train)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, y_pred, y_true, y_proba = test(self.model, self.testloader, self.criterion, device=self.device)
        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path + f"_client_{self.cid}.png",
                            self.classes)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path + f"_client_{self.cid}.png",
                         len(self.classes))
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

    def save_model(self, model_path="global_model.pth"):
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition_id", required=True, type=int,
                        help="Partition of the dataset divided into 3 iid partitions created artificially.")
    parser.add_argument("--num_clients", required=True, type=int, help="Number of clients to simulate.")

    parser.add_argument('--server_address', type=str, default='[::]:8080')
    parser.add_argument('--max_epochs', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cpu', type=str,
                        help="- Choice of the device between cpu and gpu "
                             "(cuda if compatible Nvidia and mps if on mac\n"
                             "- The choice the output may be cpu even if you choose the gpu if the latter isn't "
                             "compatible")
    parser.add_argument('--dataset', default='cifar', help="choice of the dataset (cifar by default)")
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to the training data')
    parser.add_argument('--arch', type=str, default='CNNCifar',
                        help='Model architecture defined  in model_builder.py (default: simpleNet)')
    parser.add_argument('--model_save', type=str, default='', help='Path to save the central model')
    parser.add_argument('--split', default=10, type=int,
                        help='ratio (in percent) of the training dataset that will be used for the test (default : 10)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate for the central model (default : 0.001)')

    parser.add_argument('--dp', default=False, required=False, action='store_true', dest="dp",
                        help='True if we want to use the differential privacy (by default : False)')

    parser.add_argument('--choice_loss', type=str, default="cross_entropy",
                        help='Loss function to use (default : cross_entropy)')
    parser.add_argument('--choice_optimizer', type=str, default="Adam",
                        help='Optimizer to use (default : Adam)')
    parser.add_argument('--choice_scheduler', default='StepLR', type=str,
                        help='Scheduler to use (default : StepLR)')
    parser.add_argument('--early_stopping', default=False, required=False, action='store_true',
                        dest="early_stopping",
                        help='True if we want to use the early stopping (by default : False)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for the early stopping (default : 5)')

    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Path to save the confusion matrix')
    parser.add_argument('--roc_path', type=str, default=None,
                        help='Path to save the roc figures')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save the results')

    parser.add_argument('--epsilon', default=50.0, type=float,
                        help='You can play around with the level of privacy.'
                             'epsilon : Smaller epsilon means more privacy,'
                             'more noise -- and hence lower accuracy.'
                             'One useful technique is to pre-train a model on public (non-private) data,'
                             'before completing the training on the private training data.'
                        )

    args = parser.parse_args()

    print(f"Number of clients: {args.num_clients}")

    train_loader, test_loader, list_classes = load_data(partition_id=args.partition_id, num_clients=args.num_clients,
                                                        name_dataset=args.dataset, data_root=args.data_path,
                                                        resize=32 if args.dataset == 'alzheimer' else None,
                                                        batch_size=args.batch_size)

    print("start client", args.partition_id, train_loader)
    client = FlowerClient(
        cid=args.partition_id,
        classes=list_classes,
        trainloader=train_loader,
        testloader=test_loader,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        model_choice=args.arch,
        device=args.device,
        dp=args.dp,
        delta=1e-5,  # The target δ of the (ϵ,δ)-differential privacy guarantee.
        epsilon=args.epsilon,
        max_grad_norm=1.2,  # Tuning max_grad_norm is very important
        lr=args.lr,
        choice_loss=args.choice_loss,
        choice_optimizer=args.choice_optimizer,
        choice_scheduler=args.choice_scheduler,
        save_results=args.save_results,
        matrix_path=args.matrix_path,
        roc_path=args.roc_path
        ).to_client()

    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )

# Run the client with the following command:
# python cfl/client.py --partition_id 0 --num_clients 3  --batch_size 32 --dataset alzheimer --arch mobilenet --max_epochs 2 --device mps --data_path ./data/  --save_results ./results/FL/ --matrix_path confusion_matrix --roc_path roc
