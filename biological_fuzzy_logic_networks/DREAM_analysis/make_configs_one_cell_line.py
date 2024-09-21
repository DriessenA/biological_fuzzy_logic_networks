from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import os

assert ParameterGrid
assert ParameterSampler


def create_and_save_configs(sampled_params, base_config, i):
    config = base_config.copy()
    for key, value in sampled_params.items():
        config[key] = value

    config["data_file"] = sampled_params["cell_lines"]
    config["output_dir"] = f"{base_config['output_dir']}{i}/"
    config["checkpoint_path"] = f"{base_config['checkpoint_path']}{i}/"
    config["param_setting"] = i

    try:
        os.mkdir(config["output_dir"])
    except FileExistsError:
        print("Directory already exists")

    config[
        "data_file"
    ] = f"/path/to/data/{sampled_params['cell_lines']}.csv"

    config["learning_rate"] = sampled_params["learning_rate"]
    config["n_epochs"] = sampled_params["n_epochs"]
    config["batch_size"] = sampled_params["batch_size"]
    config["scale_type"] = sampled_params["normalisation"]

    with open(f"{config['output_dir']}{i}_config.json", "w") as fp:
        json.dump(config, fp)

    with open(f"{base_config['output_dir']}/Configs/{i}_config.json", "w") as fp:
        json.dump(config, fp)


def main(base_config, param_grid):
    param_list = list(ParameterGrid(param_grid))

    for i, params in enumerate(param_list):
        create_and_save_configs(sampled_params=params, base_config=base_config, i=i)


if __name__ == "__main__":
    param_grid = {
        "cell_lines": ["BT20"],
        "learning_rate": [0.0005, 0.001, 0.005],
        "n_epochs": [5, 10, 20],
        "batch_size": [64, 128, 256],
        "normalisation": ["minmax", "quantile", "clipping"],
    }

    base_config = {
        "pkn_sif": "/path/to/PKN.sif",
        "network_class": "DREAMBioFuzzNet", # Don't change
        "data_file": "",
        "output_dir": "/path/to/dir/",
        "time_point": 9,
        "non_marker_cols": ["treatment", "cell_line", "time"],
        "treatment_col_name": "treatment",
        "sample_n_cells": False,
        "filter_starved_stim": False,
        "add_root_values": True,
        "root_nodes": ["EGFR"],
        "input_value": 1,
        "train_treatments": None,
        "valid_treatments": None,
        "train_cell_lines": None,
        "valid_cell_lines": None,
        "convergence_check": False,
        "replace_zero_inputs": 1e-9,
        "inhibition_value": 1.0,
        "learning_rate": 1e-3,
        "n_epochs": 20,
        "batch_size": 300,
        "tensors_to_cuda": True,
        "checkpoint_path": "/path/to/checkpoint/dir/",
        "experiment_name": "OneCellLine_MEK_FAK_ERK",
        "optimizer": "SGD",
    }

    main(base_config=base_config, param_grid=param_grid)
