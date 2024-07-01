import argparse
import warnings
from collections import OrderedDict
from typing import List
import numpy as np
import sys
import os

import flwr as fl
import torch
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from going_modular.data_setup import load_data
from going_modular.utils import (choice_device, fct_loss, choice_optimizer_fct, choice_scheduler_fct, save_matrix,
                                 save_roc, save_graphs)
from going_modular.engine import train, test
from going_modular.model import Net
from going_modular.security import PrivacyEngine


warnings.filterwarnings("ignore", category=UserWarning)


def get_parameters2(net) -> List[np.ndarray]:
    """
    Get the parameters of the network.
    :param net: Network to get the parameters (weights and biases)
    :return: list of parameters (weights and biases) of the network
    """

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, classes, trainloader, valloader, testloader, batch_size, epochs=1, model_choice="simplenet",
                 dp=True, delta=1e-5, epsilon=0.5, max_grad_norm=1.2,
                 device="gpu", lr=0.001, choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None,
                 save_results=None, matrix_path=None, roc_path=None, input_channels=3):
        self.cid = cid
        self.classes = classes
        self.device = choice_device(device)
        self.model = Net(num_classes=len(self.classes),
                         input_channels=input_channels, arch=model_choice).to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.epochs = epochs
        self.batch_size = batch_size

        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path

        self.criterion = fct_loss(choice_loss)
        self.optimizer = choice_optimizer_fct(self.model, choice_optim=choice_optimizer, lr=lr, weight_decay=1e-6)
        if choice_scheduler:
            self.scheduler = choice_scheduler_fct(self.optimizer, choice_scheduler=choice_scheduler,
                                                  step_size=10, gamma=0.1)
        else:
            self.scheduler = None

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

        results = train(self.cid, self.model, self.trainloader, self.valloader, epochs=self.epochs,
                        loss_fn=self.criterion, optimizer=self.optimizer, scheduler=self.scheduler, device=self.device,
                        dp=self.dp, delta=self.delta, max_physical_batch_size=int(self.batch_size / 4),
                        privacy_engine=self.privacy_engine)

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
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path,  # + f"_client_{self.cid}.png",
                            self.classes)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path,  # + f"_client_{self.cid}.png",
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

    parser.add_argument('--server_address', type=str, default='[::]:8050')
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

    if args.dataset.lower() == 'mnist':
        args.input_channels = 1
    else:
        args.input_channels = 3

    print(f"Number of clients: {args.num_clients}")

    train_loader, val_loader, test_loader, list_classes = load_data(partition_id=args.partition_id,
                                                                    num_clients=args.num_clients,
                                                                    name_dataset=args.dataset, data_root=args.data_path,
                                                                    resize=32 if args.dataset == 'alzheimer' else None,
                                                                    batch_size=args.batch_size)

    print("start client", args.partition_id, train_loader)
    client = FlowerClient(
        cid=args.partition_id,
        classes=list_classes,
        trainloader=train_loader,
        valloader=val_loader,
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
        roc_path=args.roc_path,
        input_channels=args.input_channels
        ).to_client()

    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )

# Run the client with the following command:
# python cfl/client.py --partition_id 0 --num_clients 3  --batch_size 32 --dataset alzheimer --arch mobilenet --max_epochs 2 --device mps --data_path ./data/  --save_results ./results/FL/ --matrix_path confusion_matrix --roc_path roc
