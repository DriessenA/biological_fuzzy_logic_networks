# Imports
import torch
import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np

import biological_fuzzy_logic_networks.biofuzznet as biofuzznet


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Relative path to the SIF file containing the network to use for simulation",
)
parser.add_argument(
    "--seed",
    type=str,
    required=False,
    help="Seed for reproducible simulations",
    default=2,
)
parser.add_argument(
    "--outputfolder",
    type=str,
    required=True,
    help="Relative path to the folder where the simulation output will be saved. Should end with a backslash. Folder will be created if it does not exist",
)
parser.add_argument(
    "--train",
    type=int,
    help="Number of cells in the training set. Default = 3000",
    default=3000,
)
parser.add_argument(
    "--test",
    type=int,
    help="Number of cells in the test set. Default = 600.",
    default=600,
)
parser.add_argument(
    "--num_gates_max",
    type=int,
    required=True,
    help="Number of gates to randomy assign at maximum. Will create 10 random gate assigments for each gate number.",
)
args = parser.parse_args()

# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")

if args.seed is not None:
    torch.manual_seed(args.seed)


for fold in tqdm(range(10)):
    # Generate the data using a continuous simulation
    network_true = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.input)
    network_true.initialise_random_truth_and_output(args.train + args.test)
    tot_num_gates = len(network_true.fuzzy_nodes)
    if not os.path.exists(f"{args.outputfolder}/CV_experiment_{fold}"):
        os.mkdir(f"{args.outputfolder}/CV_experiment_{fold}")
        print(
            f"Directory {args.outputfolder}/CV_experiment_{fold} did not exist and was created"
        )
    else:
        print(
            f"Directory {args.outputfolder}/CV_experiment_{fold} already exists. File will be overwritten"
        )

    for num_gates in range(1, args.num_gates_max + 1):
        choices = np.random.choice(tot_num_gates, num_gates, replace=False)
        node_names = []
        gate_id = 0
        for node in network_true.fuzzy_nodes:
            if gate_id in choices:
                coin_toss = torch.bernoulli(torch.tensor(0.5))
                node_names.append(node)
                if coin_toss:
                    network_true.nodes[node]["node_type"] = "logic_gate_OR"
                else:
                    network_true.nodes[node]["node_type"] = "logic_gate_AND"
            gate_id += 1
        print("Gates were randomly assigned.")
        pickle.dump(
            network_true,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/model_structure_data_{num_gates}.p",
                "wb",
            ),
        )

        # Set ground truth for root nodes
        for n in network_true.root_nodes:
            network_true.nodes()[n]["output_state"] = network_true.nodes()[n][
                "ground_truth"
            ]
        inputs = {
            n: network_true.nodes()[n]["output_state"] for n in network_true.root_nodes
        }

        # Simulate
        network_true.sequential_update(network_true.root_nodes)
        # The result of the simulation will be our ground truth
        ground_truth = {
            n: network_true.nodes()[n]["output_state"]
            for n in network_true.biological_nodes
        }

        # Define a train and test set
        train_input = {key: val[0 : args.train] for key, val in inputs.items()}
        test_input = {
            key: val[args.train : args.train + args.test] for key, val in inputs.items()
        }

        train_ground_truth = {
            key: val[0 : args.train] for key, val in ground_truth.items()
        }
        test_ground_truth = {
            key: val[args.train : args.train + args.test]
            for key, val in ground_truth.items()
        }

        # Save the datasets
        pickle.dump(
            train_input,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/train_input_{num_gates}.p",
                "wb",
            ),
        )
        pickle.dump(
            test_input,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/test_input_{num_gates}.p",
                "wb",
            ),
        )
        pickle.dump(
            train_ground_truth,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/train_ground_truth_{num_gates}.p",
                "wb",
            ),
        )
        pickle.dump(
            test_ground_truth,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/test_ground_truth_{num_gates}.p",
                "wb",
            ),
        )
        pickle.dump(
            node_names,
            open(
                f"{args.outputfolder}/CV_experiment_{fold}/node_names_{num_gates}.p",
                "wb",
            ),
        )
