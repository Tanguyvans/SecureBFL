import os

import torch
import json

from going_modular.data_setup import load_dataset
from going_modular.utils import choice_device, np
from flowerclient import FlowerClient


# %%
def get_model_files(dir_model, training_approach="CFL"):
    all_files = os.listdir(dir_model)
    model_files = [file for file in all_files if file.endswith('.npz') or (file.endswith('.pth') and training_approach == "scratch")]
    return model_files


# %%
if __name__ == '__main__':
    # %%
    # récupérer la config du fichier json: results/CFL/config.json
    training_approach = "BFL"  # "CFL" # "BFL"  # scratch
    matrix_path = "matrix"
    roc_path = "roc"
    device = "mps"
    with open(f"results/{training_approach}/config.json", "r") as f:
        config = json.load(f)['settings']
    # if 'settings' in config:
    #    config = config['settings']

    model_list = get_model_files(config['save_model'], training_approach)

    # Set device
    device = choice_device(device)

    # %% Load the test dataset

    _, _, node_test_sets, classes = load_dataset(config["length"],
                                                 config['name_dataset'],
                                                 config["data_root"],
                                                 1,
                                                 1)

    x_test, y_test = node_test_sets[0]

    # %%
    flower_client = FlowerClient.node(
        x_test=x_test,
        y_test=y_test,
        batch_size=config['batch_size'],
        model_choice=config['arch'],
        classes=classes,
        choice_loss=config['choice_loss'],
        choice_optimizer=config['choice_optimizer'],
        choice_scheduler=config['choice_scheduler'],
        save_figure=config['save_results'],
        matrix_path=matrix_path,
        roc_path=roc_path,
        pretrained=config['pretrained'],
    )

    # %%
    evaluation = []

    for model_file in model_list:
        if model_file.endswith('.npz'):
            loaded_weights_dict = np.load(config['save_model'] + model_file, allow_pickle=True)
        else:
            # Torch model
            loaded_weights_dict = torch.load(config['save_model'] + model_file)

        loaded_weights = [val for key, val in loaded_weights_dict.items() if 'len_dataset' not in key]

        # Evaluate the model
        metrics = flower_client.evaluate(loaded_weights, {'name': 'global_test_model_file_'})

        if model_file[0:2] == "n1" or training_approach == "scratch":
            model_file = model_file[2:]

        evaluation.append((model_file, metrics['test_loss'], metrics['test_acc']))

    # %%
    os.makedirs(config['save_results'], exist_ok=True)
    if training_approach == "scratch":
        evaluation.sort(key=lambda x: int(x[0].split('.')[0].split("_")[-1]))

    else:
        evaluation.sort(key=lambda x: int(x[0][1:].split('.')[0]))

    with open(config['save_results'] + "evaluation.txt", "w") as f:
        for model_file, loss, acc in evaluation:
            f.write(f"{model_file}: {loss}, {acc} \n")
