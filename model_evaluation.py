import os

from going_modular.data_setup import load_dataset
from going_modular.utils import choice_device, fct_loss, np
from flowerclient import FlowerClient


def get_model_files(directory):
    all_files = os.listdir(directory)
    model_files = [file for file in all_files if file.endswith('.npz')]
    return model_files


if __name__ == '__main__':
    arch = 'CNNCifar'  # "CNNCifar"
    name_dataset = 'cifar'  # "cifar"
    data_root = "data/"
    type_arch = "BFL"
    directory = f'models/'  # Update this path
    save_results = f"results/{type_arch}/"
    matrix_path = "matrix"
    roc_path = "roc"
    device = "mps"
    choice_loss = "cross_entropy"
    batch_size = 32
    model_list = get_model_files(directory)

    # Set device
    device = choice_device(device)

    # Load the test dataset

    _, _, node_test_sets, classes = load_dataset(32 if name_dataset == 'alzheimer' else None,
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
        model_choice=arch,
        classes=classes,
        choice_loss="cross_entropy",
        choice_optimizer="SGD",
        choice_scheduler=None,
        save_results=save_results,
        matrix_path=matrix_path,
        roc_path=roc_path
    )

    evaluation = []

    for model_file in model_list:
        loaded_weights_dict = np.load(directory + model_file, allow_pickle=True)
        loaded_weights = [val for key, val in loaded_weights_dict.items() if 'len_dataset' not in key]

        # Evaluate the model
        metrics = flower_client.evaluate(loaded_weights, {'name': 'global_test_model_file_'})

        if model_file[0:2] == "n1":
            model_file = model_file[2:]

        evaluation.append((model_file, metrics['test_loss'], metrics['test_acc']))

    evaluation.sort(key=lambda x: int(x[0][1:].split('.')[0]))
    with open(save_results + "evaluation.txt", "w") as file:
        for model_file, loss, acc in evaluation:
            file.write(f"{model_file}: {loss}, {acc} \n")


