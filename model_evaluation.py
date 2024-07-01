import os

from going_modular.data_setup import load_dataset
from going_modular.model import Net
from going_modular.utils import choice_device, fct_loss, np
from flowerclient import FlowerClient


def get_model_files(directory):
    all_files = os.listdir(directory)
    model_files = [file for file in all_files if file.endswith('.npz')]
    return model_files


if __name__ == '__main__':
    arch = 'CNNMnist'  # "CNNCifar"
    name_dataset = 'mnist'  #"cifar"
    data_root = "data/"
    directory = 'models/'  # Update this path
    device = "mps"
    choice_loss = "cross_entropy"
    batch_size = 32
    model_list = get_model_files(directory)

    # Set device
    device = choice_device(device)

    # Load the test dataset

    _, _, node_test_sets, classes = load_dataset(32 if arch == 'alzheimer' else None,
                                                 name_dataset,
                                                 data_root,
                                                 1,
                                                 1)

    x_test, y_test = node_test_sets[0]

    # Define the loss function
    criterion = fct_loss(choice_loss)

    flower_client = FlowerClient.node(
        x_test=x_test,
        y_test=y_test,
        batch_size=batch_size,
        dp=False,
        name_dataset=name_dataset,
        model_choice=arch,
        classes=classes,
        choice_loss="cross_entropy",
        choice_optimizer="Adam",
        choice_scheduler=None
    )

    for model_file in model_list:
        loaded_weights_dict = np.load(directory+model_file)
        loaded_weights = [loaded_weights_dict[f'param_{i}'] for i in range(len(loaded_weights_dict)-1)]

        # Evaluate the model
        metrics = flower_client.evaluate(loaded_weights, {})

        print(metrics)

        # with open("evaluation.txt", "a") as file:
        #     file.write(f"{model_file}: {metrics['loss']}, {metrics['accuracy']} \n")
