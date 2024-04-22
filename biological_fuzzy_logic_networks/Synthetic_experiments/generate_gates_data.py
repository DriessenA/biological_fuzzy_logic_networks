# Imports
import torch
import os
import json
import click
import pickle
import numpy as np
import pandas as pd

from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet


def check_create_or_warn_folder(dir):
    # Create the output folder
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directory {dir} did not exist and was created")
    else:
        print(f"Directory {dir} already exists. File will be overwritten")


def generate_gate_datasets(
    pkn_path: str,
    train_size: int,
    test_size: int,
    chosen_gates_idx: int,
    **extras,
):

    # Initialize network
    teacher_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    changed_gates = []
    for gate_idx, node in enumerate(teacher_network.fuzzy_nodes):
        if gate_idx in chosen_gates_idx:
            changed_gates.append(node)
            coin_toss = torch.bernoulli(torch.tensor(0.5))
            if coin_toss:
                teacher_network.nodes[node]["node_type"] = "logic_gate_OR"
            else:
                teacher_network.nodes[node]["node_type"] = "logic_gate_AND"

    # Simulate
    # INHIBITION INPUTS
    no_inhibition = {k: torch.ones(train_size) for k in teacher_network.nodes}
    no_inhibition_test = {k: torch.ones(test_size) for k in teacher_network.nodes}

    with torch.no_grad():
        teacher_network.initialise_random_truth_and_output(train_size)
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition
        )
        train_true = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        train_input = pd.DataFrame(
            {
                k: v.numpy()
                for k, v in teacher_network.output_states.items()
                if k in teacher_network.root_nodes
            }
        )

    train_true_df = pd.DataFrame(train_true)
    train_input_df = pd.DataFrame(train_input)

    with torch.no_grad():
        # Generate test data without perturbation
        teacher_network.initialise_random_truth_and_output(test_size)
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )
        test_true = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        test_input = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

    test_true_df = pd.DataFrame({k: v.numpy() for k, v in test_true.items()})
    test_input_df = pd.DataFrame({k: v.numpy() for k, v in test_input.items()})

    return (
        train_input_df,
        train_true_df,
        test_input_df,
        test_true_df,
        teacher_network,
        changed_gates,
    )


def repeated_data_generation(
    n_max_gates: int,
    n_repeats: int,
    out_dir: str,
    pkn_path: str,
    train_size: int,
    test_size: int,
    **extras,
):

    for n_gates in range(n_max_gates + 1):
        print(f"Simulating {n_gates}")
        for repeat in range(n_repeats):
            exp_dir = os.path.join(out_dir, f"{n_gates}_gates_{repeat}_repeat_")

            chosen_gates_idx = np.random.choice(n_max_gates, n_gates, replace=False)

            (
                train_input_df,
                train_true_df,
                test_input_df,
                test_true_df,
                model,
                changed_gates,
            ) = generate_gate_datasets(
                chosen_gates_idx=chosen_gates_idx,
                pkn_path=pkn_path,
                train_size=train_size,
                test_size=test_size,
            )

            model_state_dict, model_gate_dict = model.get_checkpoint(save_gates=True)
            torch.save(
                {
                    "model_state_dict": model_state_dict,
                    "model_gate_dict": model_gate_dict,
                },
                f"{exp_dir}model_for_simulation.pt",
            )

            train_input_df.to_csv(exp_dir + "train_input.csv", index=False)
            train_true_df.to_csv(exp_dir + "train_true.csv", index=False)
            test_input_df.to_csv(exp_dir + "test_input.csv", index=False)
            test_true_df.to_csv(exp_dir + "test_true.csv", index=False)
            pickle.dump(
                changed_gates,
                open(
                    f"{exp_dir}changed_gates.p",
                    "wb",
                ),
            )


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    torch.manual_seed(config["seed"])

    check_create_or_warn_folder(config["out_dir"])

    repeated_data_generation(**config)


if __name__ == "__main__":
    main()
