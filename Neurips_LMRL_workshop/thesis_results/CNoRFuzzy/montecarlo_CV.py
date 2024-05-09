import warnings

warnings.filterwarnings("ignore")

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from tqdm import tqdm
from datetime import datetime
import pickle

import biological_fuzzy_logic_networks.biofuzznet as biofuzznet
import biological_fuzzy_logic_networks.biomixnet as biomixnet
import biological_fuzzy_logic_networks.biofuzzdataset as biofuzzdataset
import biological_fuzzy_logic_networks.Hill_function as Hill_function
import biological_fuzzy_logic_networks.utils as utils

import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to input model")
parser.add_argument("--data", type=str, required=True, help="Path to data")
parser.add_argument(
    "--foldnum", type=int, required=True, help="Number of folds for the MC simulation"
)
parser.add_argument("--type", type=str, required=True, help="mix of fuzz")
parser.add_argument("--outputfolder", type=str, required=True, help="Path to output")
parser.add_argument(
    "--epochs", type=int, required=True, help="Number of epochs for optimisation"
)
parser.add_argument(
    "--learningrate", type=float, required=True, help="Learning rate for optimisation"
)
parser.add_argument(
    "--batchsize", type=int, required=True, help="Batch size for optimisation"
)
parser.add_argument(
    "--rounds",
    type=int,
    required=True,
    help="Number of rounds for optimisation. The model is optimised for a total of epochs*round epochs",
)
args = parser.parse_args()
# See if output folder exists
# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")

# Load the data
simulated_data = pd.read_csv(args.data, index_col=0)
# Set the seed
rng = np.random.default_rng(2)
# Prepare the results dataframe
RMSE_df = pd.DataFrame(columns=simulated_data.columns)
test_set_df = pd.DataFrame(columns=["test_1", "test_2", "test_3", "test_4", "test_5"])
if args.type == "fuzz":
    BFZ = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.model)
elif args.type == "mix":
    BFZ = biomixnet.BioMixNet.build_BuiMixNet_from_file(args.model)
else:
    ValueError(f"Unknown type {args.type}")
BFZ_params = utils.obtain_params(BFZ)[0]
params_df = pd.DataFrame(BFZ_params)

# X-folds cross validation
for fold in range(args.foldnum):
    print(f"Currently running fold {fold}")
    if args.type == "fuzz":
        BFZ = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.model)
    elif args.type == "mix":
        BFZ = biomixnet.BioMixNet.build_BuiMixNet_from_file(args.model)
    else:
        ValueError(f"Unknown type {args.type}")

    # Generate the data
    if len(simulated_data) == 22:  # Liver dataset
        input_col = ["IGF1", "IL1a", "IL6", "LPS", "TGFa", "TNFa"]
        core_train = [1, 3, 4, 5, 6, 7, 8]
        train_rows = list(rng.choice([i for i in range(9, 23)], 9, replace=False))
        test_rows = [i for i in range(9, 23) if i not in train_rows and i not in [2]]
        train_rows = train_rows + core_train
        GT_train = {
            c: torch.tensor(simulated_data.loc[train_rows, c].values)
            for c in simulated_data.columns
        }
        GT_test = {
            c: torch.tensor(simulated_data.loc[test_rows, c].values)
            for c in simulated_data.columns
        }
        input_train = {c: GT_train[c] for c in input_col}
        input_test = {c: GT_test[c] for c in input_col}
    elif len(simulated_data) == 16:  # DREAM dataset
        input_col = ["tgfa", "igf1", "il1a", "tnfa"]
        core_train = [1, 2, 3, 4, 5]
        train_rows = list(rng.choice([i for i in range(6, 16)], 5, replace=False))
        test_rows = [i for i in range(6, 16) if i not in train_rows]
        train_rows = train_rows + core_train
        print(train_rows)
        print(test_rows)
        GT_train = {
            c: torch.tensor(simulated_data.loc[train_rows, c].values)
            for c in simulated_data.columns
        }
        GT_test = {
            c: torch.tensor(simulated_data.loc[test_rows, c].values)
            for c in simulated_data.columns
        }
        input_train = {c: GT_train[c] for c in input_col}
        input_test = {c: GT_test[c] for c in input_col}
    else:
        ValueError("Unknown datafile. Cannot generate CV data")

    # Optimise
    try:
        all_losses = []
        losses = []
        loss = []
        test_loss = []
        for i in range(args.rounds):
            losses_i = BFZ.conduct_optimisation(
                input=input_train,
                ground_truth=GT_train,
                test_input=input_test,
                test_ground_truth=GT_test,
                epochs=args.epochs,
                learning_rate=args.learningrate,
                batch_size=args.batchsize,
            )
            losses.append(losses_i)
            plot_loss = sns.lineplot(data=losses_i, x="time", y="loss", hue="phase")
            plot_fig = plot_loss.get_figure()
            plot_fig.savefig(
                f"{args.outputfolder}/fold_{fold}_intermediate_loss_{i}.png"
            )

        # Concatenate the losses of the iterative runs
        # all_losses = pd.concat(losses)
        # all_losses.reset_index(inplace=True)
        # all_losses

        # Plot the losses to check optimisation happened correctly
        # plot_loss = sns.lineplot(data=all_losses, x="time", y="loss", hue="phase")
        # plot_fig = plot_loss.get_figure()
        # plot_fig.savefig(f"{args.outputfolder}/total_loss_CV{fold}.png")

        # Generate the measures that will be useful
        # Simulate the BFZ on the test set
        BFZ.set_network_ground_truth(input_test)
        BFZ.sequential_update(BFZ.root_nodes)
        # Compute the test error and save it into the dataframe
        output_RMSE = utils.compute_RMSE_outputs(model=BFZ, ground_truth=GT_test)
        RMSE = pd.Series(output_RMSE)
        RMSE_df = RMSE_df.append(RMSE, ignore_index=True)

        # Save the transfer edges parameters
        BFZ_params = utils.obtain_params(BFZ)[0]
        params = pd.Series(BFZ_params)
        params_df = params_df.append(params, ignore_index=True)

        # Save the input combinations that we used for testing
        input_combi = pd.Series(test_rows)
        input_combi.index = ["test_1", "test_2", "test_3", "test_4", "test_5"]
        test_set_df = test_set_df.append(input_combi, ignore_index=True)

    except Exception as e:
        print(f"Fold {fold} was interrupted because of error {e}")
        with open(f"{args.outputfolder}/eror_fold_{fold}.txt", "w") as f:
            f.write(f"Fold {fold} failed because of exception")
            f.write("\n")
            f.write(f"{e}")
            f.write(f"The test rows of this fold were: {test_rows}")

    # Save the data we will need later
    print(fold)

with open(f"{args.outputfolder}/hyperparameters.txt", "w") as f:
    f.write(
        f"model: {args.model}, epochs: {args.epochs}, batchsize: {args.batchsize}, rounds: {args.rounds}, learningrate: {args.learningrate}, type: {args.type}, folds: {args.foldnum}"
    )
    f.close()
RMSE_df.to_csv(f"{args.outputfolder}/RMSE_df.csv")
params_df.to_csv(f"{args.outputfolder}/params_df.csv")
test_set_df.to_csv(f"{args.outputfolder}/test_set_df.csv")
