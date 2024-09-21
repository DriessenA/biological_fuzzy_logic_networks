# Imports
import torch
import os
import pickle
import click
import json
import pandas as pd
import numpy as np

from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import (
    DREAMBioFuzzNet,
    DREAMBioMixNet,
)
import biological_fuzzy_logic_networks.utils as utils


def load_data(exp_dir: str, val_frac: float):
    node_names = pickle.load(open(f"{exp_dir}changed_gates.p", "rb"))
    train_input_df = pd.read_csv(f"{exp_dir}train_input.csv")
    train_true_df = pd.read_csv(f"{exp_dir}train_true.csv")
    test_input_df = pd.read_csv(f"{exp_dir}test_input.csv")
    test_true_df = pd.read_csv(f"{exp_dir}test_true.csv")

    # Simulation model checkpoint
    bfn_ckpt = torch.load(f"{exp_dir}model_for_simulation.pt")

    test_size = len(test_true_df)
    test_input_dict = {
        c: torch.Tensor(np.array(test_input_df[c])) for c in test_input_df.columns
    }
    test_dict = {
        c: torch.Tensor(np.array(test_true_df[c])) for c in test_true_df.columns
    }

    # Train student on unperturbed training data
    # Split train data in training and validation data
    val = train_true_df.sample(frac=val_frac)
    train = train_true_df.drop(val.index, axis=0)
    train_size = len(train)
    val_size = len(val)

    train_dict = {c: torch.Tensor(np.array(train[c])) for c in train.columns}
    val_dict = {c: torch.Tensor(np.array(val[c])) for c in val.columns}

    # Same input as teacher:
    train_input = train_input_df.iloc[train.index, :]
    val_input = train_input_df.drop(train.index, axis=0)

    train_input_dict = {
        c: torch.Tensor(np.array(train_input[c])) for c in train_input.columns
    }
    val_input_dict = {
        c: torch.Tensor(np.array(val_input[c])) for c in val_input.columns
    }

    # Data should have root nodes and non-root nodes
    val_dict.update(val_input_dict)
    train_dict.update(train_input_dict)
    test_dict.update(test_input_dict)

    # Inhibitors
    train_inhibitors = {c: torch.ones(train_size) for c in train_dict.keys()}
    val_inhibitors = {c: torch.ones(val_size) for c in val_dict.keys()}
    test_inhibitors = {c: torch.ones(test_size) for c in test_dict.keys()}

    return (
        node_names,
        train_size,
        train_input_dict,
        train_dict,
        train_inhibitors,
        val_size,
        val_input_dict,
        val_dict,
        val_inhibitors,
        test_size,
        test_input_dict,
        test_dict,
        test_inhibitors,
        bfn_ckpt,
    )


def fit_to_data(exp_dir, pkn_path, BFN_training_params, val_frac):
    (
        node_names,
        train_size,
        train_input_dict,
        train_dict,
        train_inhibitors,
        val_size,
        val_input_dict,
        val_dict,
        val_inhibitors,
        test_size,
        test_input_dict,
        test_dict,
        test_inhibitors,
        bfn_checkpoint,
    ) = load_data(exp_dir=exp_dir, val_frac=val_frac)

    bfn = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(pkn_path)
    bfn.load_from_checkpoint(
        bfn_checkpoint["model_state_dict"], bfn_checkpoint["model_gate_dict"]
    )
    bmn = DREAMBioMixNet.build_DREAMBioMixNet_from_file(pkn_path)

    # Setup gates for BioMixNet
    for gate in bmn.mixed_gates:
        if gate not in node_names:
            bmn.nodes()[gate]["node_type"] = bfn.nodes()[gate]["node_type"]
            del bmn.nodes()[gate]["gate"]

    bmn.initialise_random_truth_and_output(
        train_size, to_cuda=BFN_training_params["tensors_to_cuda"]
    )

    losses, curr_best_val_loss = bmn.conduct_optimisation(
        input=train_input_dict,
        ground_truth=train_dict,
        train_inhibitors=train_inhibitors,
        valid_ground_truth=val_dict,
        valid_input=val_input_dict,
        valid_inhibitors=val_inhibitors,
        **BFN_training_params,
    )

    # Get output states test set
    with torch.no_grad():
        bmn.initialise_random_truth_and_output(
            test_size, to_cuda=BFN_training_params["tensors_to_cuda"]
        )

        bmn.set_network_ground_truth(
            test_dict, to_cuda=BFN_training_params["tensors_to_cuda"]
        )

        bmn.sequential_update(
            bmn.root_nodes,
            inhibition=test_inhibitors,
            to_cuda=BFN_training_params["tensors_to_cuda"],
        )
        test_output = {
            k: v.cpu() for k, v in bmn.output_states.items() if k not in bmn.root_nodes
        }
        test_output_df = pd.DataFrame({k: v.numpy() for k, v in test_output.items()})

    return losses, bmn, test_output_df


def repeated_gate_learning(
    n_gates, n_repeats, out_dir, pkn_path, val_frac, BFN_training_params, **extras
):
    print(f"Fitting {n_gates} gates")
    for repeat in range(n_repeats):
        print(f"Repeat {repeat}")
        exp_dir = os.path.join(out_dir, f"{n_gates}_gates_{repeat}_repeat_")

        (losses, bmn, test_output_df) = fit_to_data(
            exp_dir=exp_dir,
            pkn_path=pkn_path,
            val_frac=val_frac,
            BFN_training_params=BFN_training_params,
        )
        model_state_dict, model_gate_dict = bmn.get_checkpoint(save_gates=True)
        torch.save(
            {"model_state_dict": model_state_dict, "model_gate_dict": model_gate_dict},
            f"{exp_dir}model_for_prediction.pt",
        )
        parameters_opt, n_opt, K_opt = utils.obtain_params(bmn)
        pickle.dump(
            parameters_opt,
            open(f"{exp_dir}parameters.p", "wb"),
        )
        test_output_df.to_csv(exp_dir + "test_output.csv", index=False)
        losses.to_csv(f"{exp_dir}losses.csv")

        del bmn


@click.command()
@click.argument("config_path")
@click.option("--n_gates", "-n", "n_gates", type=int)
def main(config_path, n_gates):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    repeated_gate_learning(n_gates=n_gates, **config)


if __name__ == "__main__":
    main()
