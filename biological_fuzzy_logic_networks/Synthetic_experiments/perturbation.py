from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet

# from biological_fuzzy_logic_networks.DREAM_analysis.train_network import get_environ_var
# from app_tunnel.apps import mlflow_tunnel

import torch
import numpy as np
import pandas as pd
import json
import click

# import mlflow


def run_sim_and_baselines(
    pkn_path,
    train_size,
    test_size,
    inhibited_node: str = "mek12",
    k_inhibition: float = 5.0,
    divide_inhibition: float = 10.0,
    train_frac=0.7,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
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
    untrained_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    # INHIBITION INPUTS
    no_inhibition = {k: torch.ones(train_size) for k in teacher_network.nodes}
    no_inhibition_test = {k: torch.ones(test_size) for k in teacher_network.nodes}
    perturb_inhibition = no_inhibition_test.copy()
    perturb_inhibition[inhibited_node] = torch.Tensor([divide_inhibition] * test_size)

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

    # Generate test data with perturbation
    # Introduce perturbation
    original_params = []
    for e in teacher_network.edges:
        if e[0] == inhibited_node:
            original_params.append(teacher_network.edges[e]["layer"].K)
            teacher_network.edges[e]["layer"].K = torch.nn.Parameter(
                torch.Tensor([k_inhibition])
            )

    # Generate test data with perturbation
    with torch.no_grad():
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )
        perturb_data = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        perturb_input = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

        perturb_true_df = pd.DataFrame({k: v.numpy() for k, v in perturb_data.items()})

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

    # TEACHER network with division inhibition (K back to original), same inputs
    counter = 0
    for e in teacher_network.edges:
        if e[0] == inhibited_node:
            teacher_network.edges[e]["layer"].K = original_params[counter]
            counter += 1

    with torch.no_grad():
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=perturb_inhibition
        )
        true_with_i_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        teach_div_perturb_with_i = pd.DataFrame(true_with_i_data)

    # TEACHER network with division inhibition (K back to original), random inputs
    with torch.no_grad():
        teacher_network.initialise_random_truth_and_output(test_size)
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=perturb_inhibition
        )
        true_wo_i_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        teach_div_perturb = pd.DataFrame(true_wo_i_data)

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

    # TEST student network without perturbation, random inputs
    with torch.no_grad():
        student_network.initialise_random_truth_and_output(test_size)
        student_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )

        test_random_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_random_output_df = pd.DataFrame(
            {k: v.numpy() for k, v in test_random_output.items()}
        )

    # TEST student network with perturbation, same inputs
    perturb_ground_truth = perturb_input.copy()
    perturb_ground_truth.update(perturb_data)

    with torch.no_grad():
        student_network.initialise_random_truth_and_output(test_size)
        student_network.set_network_ground_truth(perturb_ground_truth)
        student_network.sequential_update(
            teacher_network.root_nodes, inhibition=perturb_inhibition
        )

        perturb_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        perturb_output_df = pd.DataFrame(
            {k: v.numpy() for k, v in perturb_output.items()}
        )

    # TEST student network with perturbation, random inputs
    with torch.no_grad():
        untrained_network.initialise_random_truth_and_output(test_size)
        untrained_network.sequential_update(
            untrained_network.root_nodes, inhibition=perturb_inhibition
        )
        perturb_gen = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        perturb_gen_df = pd.DataFrame(perturb_gen)

    # UNTRAINED NETWORK on perturbation, same inputs
    with torch.no_grad():
        untrained_network.initialise_random_truth_and_output(test_size)
        untrained_network.set_network_ground_truth(perturb_ground_truth)
        untrained_network.sequential_update(
            teacher_network.root_nodes, inhibition=perturb_inhibition
        )
        ut_perturb_with_input = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_perturb_with_input_df = pd.DataFrame(ut_perturb_with_input)

    # UNTRAINED NETWORK on perturbation, random inputs
    with torch.no_grad():
        untrained_network.initialise_random_truth_and_output(test_size)
        untrained_network.sequential_update(
            untrained_network.root_nodes, inhibition=perturb_inhibition
        )
        ut_perturb = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }

        ut_perturb_df = pd.DataFrame(ut_perturb)

    # UNTRAINED NETWORK without perturbation, same inputs
    with torch.no_grad():
        untrained_network.initialise_random_truth_and_output(test_size)
        untrained_network.set_network_ground_truth(test_ground_truth)
        untrained_network.sequential_update(
            untrained_network.root_nodes, inhibition=no_inhibition_test
        )
        gen_with_i_test = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_test_with_i_df = pd.DataFrame(gen_with_i_test)

    # UNTRAINED NETWORK without perturbation, random inputs
    with torch.no_grad():
        untrained_network.initialise_random_truth_and_output(test_size)
        untrained_network.sequential_update(
            untrained_network.root_nodes, inhibition=no_inhibition_test
        )
        gen_test = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_test_df = pd.DataFrame(gen_test)

    pertubed_pred_data = pd.concat(
        [
            perturb_true_df,
            teach_div_perturb_with_i,
            teach_div_perturb,
            perturb_output_df,
            perturb_gen_df,
            ut_perturb_with_input_df,
            ut_perturb_df,
        ],
        keys=[
            "teacher_k_inhibition_true",
            "teacher_division_same_input",
            "teacher_division_random_input",
            "student_division_same_input",
            "student_division_random_input",
            "untrained_division_same_input",
            "untrained_division_random_input",
        ],
    )

    unpertubed_pred_data = pd.concat(
        [
            test_true_df,
            test_output_df,
            test_random_output_df,
            ut_test_with_i_df,
            ut_test_df,
        ],
        keys=[
            "teacher_no_pertrub_true",
            "student_no_perturb_same_input",
            "student_no_perturb_random_input",
            "untrained_no_perturb_same_input",
            "untrained_no_perturb_random_input",
        ],
    )

    return (
        losses,
        pertubed_pred_data,
        unpertubed_pred_data,
        student_network.get_checkpoint(),
        teacher_network.get_checkpoint(),
    )


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    losses, pertubed_data, unpertubed_data, student, teacher = run_sim_and_baselines(
        **config
    )

    losses.to_csv(f"{config['out_dir']}_losses.csv")
    pertubed_data.to_csv(f"{config['out_dir']}_perturbed.csv")
    unpertubed_data.to_csv(f"{config['out_dir']}_unperturbed.csv")

    torch.save({"model_state_dict": teacher}, f"{config['out_dir']}_teacher.pt")
    torch.save({"model_state_dict": student}, f"{config['out_dir']}_student.pt")


if __name__ == "__main__":
    main()
