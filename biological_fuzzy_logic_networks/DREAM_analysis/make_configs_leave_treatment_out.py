from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import os

assert ParameterGrid
assert ParameterSampler


def create_and_save_configs(sampled_params, base_config, i):
    config = base_config.copy()
    for key, value in sampled_params.items():
        config[key] = value

    config["output_dir"] = f"{base_config['output_dir']}{i}/"
    config["checkpoint_path"] = f"{base_config['checkpoint_path']}{i}/"
    config["param_setting"] = i

    try:
        os.mkdir(config["output_dir"])
    except FileExistsError:
        print("Directory already exists")

    config[
        "data_file"
    ] = f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/CL_incl_test/{ sampled_params['cell_lines']}.csv"

    config["test_treatments"] = sampled_params["test_treatments"]
    config["train_treatments"] = sampled_params["train_treatments"]
    if sampled_params["test_treatments"] == "imTOR":
        config["valid_treatments"] = "iPKC"
    elif sampled_params["test_treatments"] == "iPKC":
        config["valid_treatments"] = "imTOR"

    config["learning_rate"] = sampled_params["learning_rate"]
    config["n_epochs"] = sampled_params["n_epochs"]
    config["batch_size"] = sampled_params["batch_size"]
    config["inhibition_value"] = sampled_params["inhibition_value"]

    with open(f"{config['output_dir']}{i}_config.json", "w") as fp:
        json.dump(config, fp)

    with open(f"{base_config['output_dir']}/Configs/{i}_config.json", "w") as fp:
        json.dump(config, fp)


def main(base_config, param_grid):
    param_list = list(ParameterGrid(param_grid))

    for i, params in enumerate(param_list):
        create_and_save_configs(params, base_config, f"{i}")


if __name__ == "__main__":
    param_grid = {
        "cell_lines": ["BT549", "HCC1954", "OCUBM"],
        "test_treatments": ["imTOR", "iPKC"],
        "train_treatments": [["iEGFR", "iPI3K", "iMEK"]],
        # "valid_cell_lines": [],
        # "test_cell_lines": [["HCC1806", "Hs578T", "HCC1428"]],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "n_epochs": [10, 50, 100],
        "batch_size": [128, 1000, 10000],
        "inhibition_value": [1, 5, 10],
    }

    base_config = {
        "pkn_sif": "/path/to/PKN_Alice_AMPK_SMAD23_roots.sif",
        "network_class": "DREAMBioFuzzNet",
        "data_file": "",
        "output_dir": "/path/to/dir/",
        "time_point": 9,
        "non_marker_cols": ["treatment", "cell_line", "time", "cellID", "fileID"],
        "treatment_col_name": "treatment",
        "sample_n_cells": 500,
        "filter_starved_stim": True,
        "minmaxscale": True,
        "add_root_values": True,
        "root_nodes": [
            "EGFR",
            "SERUM",
        ],  # AMPK and SMAD23 added internally but not with value 1
        "input_value": 1,
        "train_treatments": None,
        "valid_treatments": None,
        "test_treatments": None,
        "train_cell_lines": None,
        "valid_cell_lines": None,
        "test_cell_lines": None,
        "convergence_check": False,
        "replace_zero_inputs": 1e-9,
        "inhibition_value": 1.0,
        "learning_rate": 1e-3,
        "n_epochs": 20,
        "batch_size": 300,
        "tensors_to_cuda": True,
        "checkpoint_path": "/path/to/save/checkpoint/",
        "experiment_name": "BFN_NT",
        "optimizer": "SGD",
    }

    main(base_config=base_config, param_grid=param_grid)
