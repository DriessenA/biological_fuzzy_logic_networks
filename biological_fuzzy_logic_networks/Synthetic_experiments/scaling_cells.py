import timeit
import torch
import json

import numpy as np
import pandas as pd

from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import DREAMBioFuzzNet


def load_data(data_dir):
    train_true = pd.read_csv(f"{data_dir}/train_true_df.csv")
    train_input = pd.read_csv(f"{data_dir}/train_input_df.csv")

    valid_true = pd.read_csv(f"{data_dir}/test_true_df.csv")
    valid_input = pd.read_csv(f"{data_dir}/test_input_df.csv")

    return train_true, train_input, valid_true, valid_input


def run_training(
    pkn_path,
    data_dir,
    n_training_cells,
    n_valid_cells,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
    **extras,
):

    student_network = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(pkn_path)

    train, train_input, val, val_input = load_data(data_dir=data_dir)

    all_train = pd.concat([train, train_input], axis=1).sample(
        n=n_training_cells, replace=False
    )
    all_val = pd.concat([val, val_input], axis=1).sample(n=n_valid_cells, replace=False)

    all_val[all_val > 1] = 1
    all_train[all_train > 1] = 1

    val_input_dict = {c: torch.Tensor(np.array(all_val[c])) for c in val_input.columns}
    train_input_dict = {
        c: torch.Tensor(np.array(all_train[c])) for c in train_input.columns
    }
    train_dict = {c: torch.Tensor(np.array(all_train[c])) for c in all_train.columns}
    val_dict = {c: torch.Tensor(np.array(all_val[c])) for c in all_val.columns}

    # Inhibitors
    train_inhibitors = {c: torch.ones(n_training_cells) for c in train_dict.keys()}
    val_inhibitors = {c: torch.ones(n_valid_cells) for c in val_dict.keys()}

    student_network.initialise_random_truth_and_output(
        n_training_cells, to_cuda=BFN_training_params["tensors_to_cuda"]
    )

    start = timeit.default_timer()
    losses, curr_best_val_loss, _ = student_network.conduct_optimisation(
        input=train_input_dict,
        ground_truth=train_dict,
        train_inhibitors=train_inhibitors,
        valid_ground_truth=val_dict,
        valid_input=val_input_dict,
        valid_inhibitors=val_inhibitors,
        **BFN_training_params,
    )
    stop = timeit.default_timer()

    return stop - start


def train_with_increasing_cell_num(
    pkn_path,
    data_dir,
    n_cells_list,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
    **extras,
):
    times = []
    for n_cells in n_cells_list:
        print(n_cells)
        time = run_training(
            pkn_path,
            data_dir,
            n_training_cells=n_cells,
            n_valid_cells=500,
            BFN_training_params=BFN_training_params,
        )

        times.append(time)

    res = pd.DataFrame({"time": times, "n_cells": n_cells_list})
    return res


if __name__ == "__main__":
    simulation_config_path = "/u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/simulation_for_scaling_config.json"
    with open(simulation_config_path, "r") as f:
        simulation_config = json.load(f)

    pkn_path = simulation_config["pkn_path"]
    data_dir = simulation_config["out_dir"]

    BFN_training_params = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
        "patience": 100,
    }
    n_cells_list = [10000000]  # [100, 1000, 10000, 100000, 1000000, 10000000]
    res = train_with_increasing_cell_num(
        pkn_path=pkn_path,
        data_dir=data_dir,
        n_cells_list=n_cells_list,
        BFN_training_params=BFN_training_params,
    )
    res.to_csv("/dccstor/ipc1/CAR/BFN/Model/Scaling/output_times.csv")
