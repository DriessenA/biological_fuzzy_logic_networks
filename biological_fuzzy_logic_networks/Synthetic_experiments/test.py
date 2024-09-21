from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet

# from biological_fuzzy_logic_networks.DREAM_analysis.train_network import get_environ_var
# from app_tunnel.apps import mlflow_tunnel

import torch
import numpy as np
import pandas as pd

# import mlflow


def run_sim_and_baselines(
    pkn_path="/dccstor/ipc1/CAR/BFN/Data/CNoRFuzzy/optimised_structure_DREAM_reduced.sif",
    train_size=24,
    test_size=5,
    train_frac=0.7,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 3,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
    **extras,
):
    teacher_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )
    student_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    # INHIBITION INPUTS
    no_inhibition = {k: torch.ones(train_size) for k in teacher_network.nodes}
    no_inhibition_test = {k: torch.ones(test_size) for k in teacher_network.nodes}

    # Generate training data without perturbation
    with torch.no_grad():
        teacher_network.initialise_random_truth_and_output(train_size)
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition
        )
        true_unperturbed_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        input_data = pd.DataFrame(
            {
                k: v.numpy()
                for k, v in teacher_network.output_states.items()
                if k in teacher_network.root_nodes
            }
        )

    train_true_df = pd.DataFrame(true_unperturbed_data)
    input_df = pd.DataFrame(input_data)

    # Generate test data without perturbation
    with torch.no_grad():
        teacher_network.initialise_random_truth_and_output(test_size)
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )
        test_data = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        test_input = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

        test_true_df = pd.DataFrame({k: v.numpy() for k, v in test_data.items()})

    # Train student on unperturbed training data
    # Split train data in training and validation data
    train = train_true_df.sample(frac=train_frac)
    val = train_true_df.drop(train.index, axis=0)

    train_dict = {c: torch.Tensor(np.array(train[c])) for c in train.columns}
    val_dict = {c: torch.Tensor(np.array(val[c])) for c in val.columns}

    # Same input as teacher:
    train_input = input_df.iloc[train.index, :]
    val_input = input_df.drop(train.index, axis=0)

    train_input_dict = {
        c: torch.Tensor(np.array(train_input[c])) for c in train_input.columns
    }
    val_input_dict = {
        c: torch.Tensor(np.array(val_input[c])) for c in val_input.columns
    }

    # Data should have root nodes and non-root nodes
    val_dict.update(val_input_dict)
    train_dict.update(train_input_dict)

    # Inhibitor
    train_inhibitors = {c: torch.ones(len(train)) for c in train_dict.keys()}
    val_inhibitors = {c: torch.ones(len(val)) for c in val_dict.keys()}

    student_network.initialise_random_truth_and_output(train_size)
    losses, curr_best_val_loss, _ = student_network.conduct_optimisation(
        input=train_input_dict,
        ground_truth=train_dict,
        train_inhibitors=train_inhibitors,
        valid_ground_truth=val_dict,
        valid_input=val_input_dict,
        valid_inhibitors=val_inhibitors,
        **BFN_training_params,
    )

    print(curr_best_val_loss)

    # TEST student without perturbation, same inputs
    test_ground_truth = test_input.copy()
    test_ground_truth.update(test_data)

    with torch.no_grad():
        student_network.initialise_random_truth_and_output(test_size)
        student_network.set_network_ground_truth(test_ground_truth)
        student_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )

        test_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_output_df = pd.DataFrame({k: v.numpy() for k, v in test_output.items()})

    return (
        losses,
        student_network.get_checkpoint(),
        teacher_network.get_checkpoint(),
    )


def main():
    losses, student, teacher = run_sim_and_baselines()


if __name__ == "__main__":
    main()
