import os
import re

import torch
import json

from going_modular.data_setup import load_dataset
from going_modular.utils import choice_device, np
from flowerclient import FlowerClient

def get_model_files(dir_model, training_approach="CFL"):
    all_files = os.listdir(dir_model)
    model_files = [file for file in all_files if file.endswith('.npz') or file.endswith('.pt') or (file.endswith('.pth') and training_approach == "scratch")]
    return model_files

def get_global_model_storage_reference(file_path):
    with open(file_path, 'r') as ff:
        content = ff.read()

    # Regex (Regular expression) to extract blocks with the global model
    pattern = r'(?s)model type:\s*global_model.*?storage reference:\s*(\S+)'

    # Search for all matches
    matches = re.findall(pattern, content)

    return matches[-1] if matches else None

if __name__ == '__main__':
    training_approach = "CFL"
    matrix_path = "matrix"
    roc_path = "roc"
    path_nodetxt = "results/BFL/node1.txt"
    dir_model = f"models/{training_approach}"
    device = "mps"
    with open(f"results/{training_approach}/config.json", "r") as f:
        config = json.load(f)['settings']
    # if 'settings' in config:
    #    config = config['settings']

    model_list = get_model_files(config['save_model'], training_approach)

    # Set device
    device = choice_device(device)

    _, _, node_test_sets, classes = load_dataset(None,
                                                 config['name_dataset'],
                                                 config["data_root"],
                                                 1,
                                                 1)

    x_test, y_test = node_test_sets[0]

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

    evaluation = []

    print(model_list)

    for model_file in sorted(model_list):
        print(f"Processing {model_file}")
        
        if model_file.endswith('.npz'):
            print(f"Loading NPZ file: {model_file}")
            loaded_weights_dict = np.load(config['save_model'] + model_file, allow_pickle=True)
            loaded_weights = [val for key, val in loaded_weights_dict.items() if 'len_dataset' not in key]
            
        elif model_file.endswith(('.pt', '.pth')):
            print(f"Loading PT/PTH file: {model_file}")
            loaded_model = torch.load(config['save_model'] + model_file)
            if isinstance(loaded_model, list):
                loaded_weights = loaded_model
            elif isinstance(loaded_model, dict):
                if 'model_state_dict' in loaded_model:
                    loaded_weights = list(loaded_model['model_state_dict'].values())
                else:
                    loaded_weights = [val for key, val in loaded_model.items() if 'len_dataset' not in key]
            else:
                print(f"Unsupported model format in {model_file}")
                continue

        print(f"Evaluating model {model_file}...")
        metrics = flower_client.evaluate(loaded_weights, {'name': 'global_test_model_file_'})
        print(f"Model evaluated - Loss: {metrics['test_loss']:.4f}, Accuracy: {metrics['test_acc']:.2f}%")

        if model_file[0:2] == "n1" or training_approach == "scratch":
            model_file = model_file[2:]

        evaluation.append((model_file, metrics['test_loss'], metrics['test_acc']))

    os.makedirs(config['save_results'], exist_ok=True)
    if training_approach == "scratch":
        evaluation.sort(key=lambda x: int(x[0].split('.')[0].split("_")[-1]))

    elif training_approach == "BFL": 
        evaluation.sort(key=lambda x: int(x[0][1:].split("_")[0]))

    else:
        evaluation.sort(key=lambda x: int(x[0][1:].split('.')[0]))

    with open(config['save_results'] + "evaluation.txt", "w") as f:
        for model_file, loss, acc in evaluation:
            f.write(f"{model_file}: {loss}, {acc} \n")

    if training_approach == "BFL":
        model_file = get_global_model_storage_reference(path_nodetxt)

        loaded_weights_dict = np.load(model_file, allow_pickle=True)
        loaded_weights = [val for key, val in loaded_weights_dict.items() if 'len_dataset' not in key]
        metrics = flower_client.evaluate(loaded_weights, {'name': 'global_test_model_file_best_'})
        print(metrics)
