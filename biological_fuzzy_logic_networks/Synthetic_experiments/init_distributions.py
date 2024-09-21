from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet

import torch
import numpy as np
import pandas as pd
import json
import click

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class Bimodal:

    def __init__(self, m1, m2, s1, s2):

        self.norm1 = torch.distributions.normal.Normal(
            torch.Tensor([m1]), torch.Tensor([s1])
        )
        self.norm2 = torch.distributions.normal.Normal(
            torch.Tensor([m2]), torch.Tensor([s2])
        )

    def sample(self, size):
        sample1 = self.norm1.sample(torch.Size([int(np.ceil(size[0] / 2))]))
        sample2 = self.norm2.sample(torch.Size([int(np.floor(size[0] / 2))]))

        return torch.cat((sample1, sample2), 0)


def student_teacher_with_init_distribution(
    pkn_path,
    train_size,
    test_size,
    dist,
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

    # Generate training data
    with torch.no_grad():
        teacher_network.initialise_random_truth_and_output(
            train_size, distribution=dist
        )
        teacher_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition
        )
        true_unperturbed_data = {
            k: v.numpy().flatten()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        input_data = pd.DataFrame(
            {
                k: v.numpy().flatten()
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
            k: v.numpy().flatten()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        test_input = {
            k: v.numpy().flatten()
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

        test_true_df = pd.DataFrame({k: v for k, v in test_data.items()})

    # Train student on training data
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

    # TEST student same inputs
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

    # TEST student network random inputs
    with torch.no_grad():
        student_network.initialise_random_truth_and_output(test_size)
        student_network.sequential_update(
            teacher_network.root_nodes, inhibition=no_inhibition_test
        )

        test_random_output = {
            k: v.numpy()
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_random_output_df = pd.DataFrame(
            {k: v for k, v in test_random_output.items()}
        )

    # UNTRAINED NETWORK same inputs
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

    # UNTRAINED NETWORK random inputs
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

    unpertubed_pred_data = pd.concat(
        [
            test_true_df,
            test_output_df,
            test_random_output_df,
            ut_test_with_i_df,
            ut_test_df,
        ],
        keys=[
            "teacher_true",
            "student_same_input",
            "student_random_input",
            "untrained_same_input",
            "untrained_random_input",
        ],
    )

    return (
        losses,
        unpertubed_pred_data,
        student_network.get_checkpoint(),
        teacher_network.get_checkpoint(),
    )


def get_prob_dist(dist_name):
    if dist_name.lower() == "normal":
        dist = Normal(torch.Tensor([0.5]), torch.Tensor([0.1]))
    elif dist_name.lower() == "uniform":
        dist = Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))
    elif dist_name.lower() == "bimodal":
        dist = Bimodal(0.2, 0.8, 0.1, 0.1)

    return dist


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    dist = get_prob_dist(config["dist_name"])
    losses, unpertubed_data, student, teacher = student_teacher_with_init_distribution(
        dist=dist, **config
    )

    losses.to_csv(f"{config['out_dir']}_losses.csv")
    unpertubed_data.to_csv(f"{config['out_dir']}_predictions.csv")

    torch.save({"model_state_dict": teacher}, f"{config['out_dir']}_teacher.pt")
    torch.save({"model_state_dict": student}, f"{config['out_dir']}_student.pt")


if __name__ == "__main__":
    main()
