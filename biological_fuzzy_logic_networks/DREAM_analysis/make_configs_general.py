from sklearn.model_selection import ParameterSampler, ParameterGrid, KFold
import numpy as np
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

    config["data_file"] = [
        f"/path/to/data/{cl}.csv"
        for cl in sampled_params["cell_lines"]
    ]
    # config["valid_cell_lines"] = [
    #     f"path/to/data/{cl}.csv"
    #     for cl in sampled_params["valid_cell_lines"]
    # ]
    config["test_cell_lines"] = [
        f"/path/to/data/{cl}.csv"
        for cl in sampled_params["test_cell_lines"]
    ]
    config["learning_rate"] = sampled_params["learning_rate"]
    config["n_epochs"] = sampled_params["n_epochs"]
    config["batch_size"] = sampled_params["batch_size"]

    with open(f"{config['output_dir']}{i}_config.json", "w") as fp:
        json.dump(config, fp)

    with open(f"{base_config['output_dir']}/Configs/{i}_config.json", "w") as fp:
        json.dump(config, fp)


def main(base_config, param_grid):
    param_list = list(ParameterGrid(param_grid))

    for i, params in enumerate(param_list):
        kf = KFold(n_splits=5, random_state=421, shuffle=True)
        for j, (train_idx, valid_idx) in enumerate(kf.split(params["cell_lines"])):
            cv_params = params.copy()
            # train_cl = list(np.array(params["cell_lines"])[train_idx])
            valid_cl = list(np.array(params["cell_lines"])[valid_idx])

            # cv_params["cell_lines"] = train_cl
            cv_params["valid_cell_lines"] = valid_cl
            create_and_save_configs(cv_params, base_config, f"{i}_{j}")


if __name__ == "__main__":
    param_grid = {
        "cell_lines": [
            [
                "BT20",
                "BT474",
                "CAL148",
                "CAL51",
                "CAL851",
                "DU4475",
                "EFM192A",
                "EVSAT",
                "HBL100",
                "HCC1599",
                "HCC2185",
                "HCC3153",
                "HDQP1",
                "JIMT1",
                "MCF7",
                "MDAMB157",
                "MDAMB175VII",
                "MDAMB415",
                "MFM223",
                "MPE600",
                "MX1",
                "OCUBM",
                "UACC812",
                "ZR7530",
            ]
        ],
        # "train_treatments": [],
        # "valid_treatments": ["iEGFR", "iMEK", "iPI3K", "iPKC"],
        # "valid_cell_lines": [],
        "test_cell_lines": [["AU565", "EFM19", "HCC2218", "LY2", "MACLS2", "MDAMB436"]],
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "n_epochs": [10, 50, 100],
        "batch_size": [1000, 5000, 10000, 50000],
    }

    base_config = {
        "pkn_sif": "/path/to/data/PKN_subnetwork.sif",
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
        "checkpoint_path": "/path/to/save/checkpoint/",
        "experiment_name": "Subnetwork",
        "optimizer": "SGD",
    }

    main(base_config=base_config, param_grid=param_grid)
