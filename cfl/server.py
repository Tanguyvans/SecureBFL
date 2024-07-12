import argparse
from collections import OrderedDict

import flwr as fl

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable
import torch.nn.functional

from logging import WARNING
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg  #, aggregate  # Aggreg functions for strategy implementation
from flwr.common import (
    Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, Scalar, logger,
    NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from going_modular.model import Net


def aggreg_fit_checkpoint(server_round, aggregated_parameters, central_model, path_checkpoint):
    if aggregated_parameters is not None:
        print(f"Saving round {server_round} aggregated_parameters...")

        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
        # Convert `List[np.ndarray]` to PyTorch`state_dict`
        params_dict = zip(central_model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        central_model.load_state_dict(state_dict, strict=True)

        # Save the model
        if path_checkpoint:
            torch.save({
                'model_state_dict': central_model.state_dict(),
            }, path_checkpoint)  # f"model_round_{server_round}.pt"

    """
    # Same function as SaveModelStrategy(fl.server.strategy.FedAvg)
    if parameters_aggregated is not None:
        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(parameters_aggregated)

        # Save aggregated_ndarrays
        print(f"Saving round {server_round} aggregated_ndarrays...")
        np.savez(f"./models/round-{server_round}-weights.npz", *aggregated_ndarrays)
    """


class SaveModelStrategy(fl.server.strategy.FedAvg):
    print("SaveModelStrategy")
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"./models/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}    


def get_on_fit_config_fn(epoch=2, lr=0.001, batch_size=32) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """
        Return training configuration dict for each round with static batch size and (local) epochs.

        Perform two rounds of training with one local epoch, increase to two local epochs afterwards.
        """
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": epoch  # 1 if server_round < 2 else epoch,
        }
        return config

    return fit_config


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
            self,
            num_classes=10,
            input_channels=3,
            arch='CNNCifar',
            device='cpu',
            model_save=None,
            **kwargs,

    ) -> None:
        super().__init__(**kwargs)
        self.central = Net(num_classes=num_classes, input_channels=input_channels, arch=arch).to(device)
        self.model_save = model_save

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # print("server configure_fit start")
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        # Custom fit config function provided
        standard_lr = args.lr
        higher_lr = 0.003
        config = {"server_round": server_round, "local_epochs": 1}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # fit_ins = FitIns(parameters, config)
        # Return client/config pairs
        fit_configurations = []
        for idx, client in enumerate(clients):
            config["learning_rate"] = standard_lr if idx < half_clients else higher_lr
            """
            Each pair of (ClientProxy, FitRes) constitutes 
            a successful update from one of the previously selected clients.
            """
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        parameters,
                        config
                    )
                )
            )
        # Successful updates from the previously selected and configured clients
        # print("server configure_fit end")
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average. (each round)"""
        # print("server aggregate_fit start")
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        aggreg_fit_checkpoint(server_round, parameters_aggregated, self.central, self.model_save)
        return parameters_aggregated, metrics_aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--num_clients", required=True, type=int, help="Number of clients to simulate.")
    parser.add_argument('--server_address', type=str, default='[::]:8050')
    parser.add_argument('--rounds', default=3, type=int, help='number of rounds (default : 3)')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the central model (default : 0.001)')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', default='cifar', help="choice of the dataset (cifar by default)")
    parser.add_argument('--arch', type=str, default='CNNCifar',
                        help='Model architecture defined  in model_builder.py (default: simpleNet)')
    parser.add_argument('--device', type=str, default='cpu', help='device to train the model (default: cpu)')

    parser.add_argument('--frac_fit', type=float, default=1.0)
    parser.add_argument('--frac_eval', type=float, default=0.5)
    parser.add_argument('--min_fit_clients', type=int, default=2)
    parser.add_argument('--min_eval_clients', type=int, default=None)
    parser.add_argument('--min_avail_clients', type=int, default=2)
    args = parser.parse_args()

    print('rounds: ', args.rounds)
    if args.dataset.lower() in ['cifar', "mnist"]:
        num_classes = 10

    elif args.dataset.lower() == 'alzheimer':
        num_classes = 4

    else:
        raise ValueError('Invalid dataset')
    
    if args.dataset.lower() == 'mnist':
        input_channels = 1
    else:
        input_channels = 3

    # Create strategy and run server
    """
    strategy = SaveModelStrategy(
        evaluate_metrics_aggregation_fn=weighted_average,
        fraction_fit=args.frac_fit,  # Train on frac_fit % clients (each round)
        min_fit_clients=args.min_fit_clients,  # Never sample less than 10 clients for training
        # Never sample less than 5 clients for evaluation
        on_fit_config_fn=get_on_fit_config_fn(
            epoch=args.max_epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
    )
    """

    strategy = FedCustom(
        fraction_fit=args.frac_fit,  # Train on frac_fit % clients (each round)
        fraction_evaluate=args.frac_eval,  # Sample frac_eval % of available clients for evaluation
        min_fit_clients=args.min_fit_clients,
        # min_evaluate_clients=args.min_eval_clients,
        min_available_clients=args.min_avail_clients,  # Wait until all min_avail_clients number of clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        # initial_parameters=ndarrays_to_parameters(get_parameters2(central)),
        # evaluate_fn=evaluate2,  # Pass the evaluation function for the server side
        on_fit_config_fn=get_on_fit_config_fn(
            epoch=args.max_epochs,
            lr=args.lr,
            batch_size=args.batch_size
        ),
        input_channels=input_channels,
        num_classes=num_classes,
        arch=args.arch,
        device=args.device,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

# Run the server with the following command:
# python cfl/server.py --num_clients 3 --rounds 3 --max_epochs 2 --batch_size 32 --dataset alzheimer --arch mobilenet --device mps
