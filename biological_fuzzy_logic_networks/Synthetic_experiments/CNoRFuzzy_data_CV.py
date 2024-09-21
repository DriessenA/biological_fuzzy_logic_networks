import torch
import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from biological_fuzzy_logic_networks import biofuzznet
from biological_fuzzy_logic_networks import biomixnet
from biological_fuzzy_logic_networks import utils
from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import DREAMBioFuzzNet


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
RMSE_df = pd.DataFrame(index=simulated_data.columns)
test_set_df = pd.DataFrame(index=["test_1", "test_2", "test_3", "test_4", "test_5"])
if args.type == "fuzz":
    BFZ = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(args.model, n=1.1, K=1)
elif args.type == "mix":
    BFZ = biomixnet.BioMixNet.build_BuiMixNet_from_file(args.model, n=1.1, K=1)
else:
    ValueError(f"Unknown type {args.type}")
BFZ_params = utils.obtain_params(BFZ)[0]
params_df = pd.DataFrame(BFZ_params).T

# X-folds cross validation
for fold in range(args.foldnum):
    print(f"Currently running fold {fold}")
    if args.type == "fuzz":
        print("Creating new model")
        BFZ = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(args.model, n=1.1, K=1)
    elif args.type == "mix":
        BFZ = biomixnet.BioMixNet.build_BuiMixNet_from_file(args.model, n=1.1, K=1)
    else:
        ValueError(f"Unknown type {args.type}")

    # Generate the data
    if len(simulated_data) == 22:  # Liver dataset
        input_col = ["IGF1", "IL1a", "IL6", "LPS", "TGFa", "TNFa"]
        simulated_data = simulated_data[
            input_col + [c for c in simulated_data if c not in input_col]
        ]
        core_train = [1, 3, 4, 5, 6, 7, 8]
        train_rows = list(rng.choice([i for i in range(9, 23)], 9, replace=False))
        test_rows = [i for i in range(9, 23) if i not in train_rows and i not in [2]]
        train_rows = train_rows + core_train
        train_data = simulated_data.loc[train_rows, :]
        test_data = simulated_data.loc[test_rows, :]
        test_inhibition = {n: torch.ones(len(test_data)) for n in test_data.columns}
        train_inhibition = {n: torch.ones(len(train_data)) for n in train_data.columns}

        scaler = MinMaxScaler()
        train_data = pd.DataFrame(
            scaler.fit_transform(train_data),
            index=train_data.index,
            columns=train_data.columns,
        )
        test_data = pd.DataFrame(
            scaler.transform(test_data),
            index=test_data.index,
            columns=test_data.columns,
        )
        train_data[train_data > 1] = 1
        test_data[test_data > 1] = 1

        GT_train = {c: torch.tensor(train_data[c].values) for c in train_data.columns}
        GT_test = {c: torch.tensor(test_data[c].values) for c in test_data.columns}
        input_train = {c: GT_train[c] for c in input_col}
        input_test = {c: GT_test[c] for c in input_col}
    elif len(simulated_data) == 16:  # DREAM dataset
        input_col = ["tgfa", "igf1", "il1a", "tnfa"]
        simulated_data = simulated_data[
            input_col + [c for c in simulated_data if c not in input_col]
        ]
        core_train = [1, 2, 3, 4, 5]
        train_rows = list(rng.choice([i for i in range(6, 16)], 5, replace=False))
        test_rows = [i for i in range(6, 16) if i not in train_rows]
        train_rows = train_rows + core_train
        train_data = simulated_data.loc[train_rows, :]
        test_data = simulated_data.loc[test_rows, :]
        test_inhibition = {n: torch.ones(len(test_data)) for n in test_data.columns}
        train_inhibition = {n: torch.ones(len(train_data)) for n in train_data.columns}

        scaler = MinMaxScaler()
        train_data = pd.DataFrame(
            scaler.fit_transform(train_data),
            index=train_data.index,
            columns=train_data.columns,
        )
        test_data = pd.DataFrame(
            scaler.transform(test_data),
            index=test_data.index,
            columns=test_data.columns,
        )
        train_data[train_data > 1] = 1
        test_data[test_data > 1] = 1

        GT_train = {c: torch.tensor(train_data[c].values) for c in train_data.columns}
        GT_test = {c: torch.tensor(test_data[c].values) for c in test_data.columns}
        input_train = {c: GT_train[c] for c in input_col}
        input_test = {c: GT_test[c] for c in input_col}
    else:
        ValueError("Unknown datafile. Cannot generate CV data")

    # Optimise
    all_losses = []
    losses = []
    loss = []
    test_loss = []
    for i in range(args.rounds):
        losses_i, best_val_loss, _ = BFZ.conduct_optimisation(
            input=input_train,
            ground_truth=GT_train,
            valid_input=input_test,
            valid_ground_truth=GT_test,
            epochs=args.epochs,
            learning_rate=args.learningrate,
            batch_size=args.batchsize,
            train_inhibitors=train_inhibition,
            valid_inhibitors=test_inhibition,
        )
        losses.append(losses_i)
        plot_loss = sns.lineplot(data=losses_i, x="time", y="loss", hue="phase")
        plot_fig = plot_loss.get_figure()
        plot_fig.savefig(f"{args.outputfolder}/fold_{fold}_intermediate_loss_{i}.png")

    # Generate the measures that will be useful
    # Simulate the BFZ on the test set
    BFZ.set_network_ground_truth(input_test)
    BFZ.sequential_update(BFZ.root_nodes, inhibition=test_inhibition)
    # Compute the test error and save it into the dataframe
    output_RMSE = utils.compute_RMSE_outputs(model=BFZ, ground_truth=GT_test)
    RMSE = pd.Series(output_RMSE)
    RMSE_df[fold] = RMSE

    # Save the transfer edges parameters
    BFZ_params = utils.obtain_params(BFZ)[0]
    params = pd.Series(BFZ_params)
    params_df[fold] = params

    # Save the input combinations that we used for testing
    input_combi = pd.Series(test_rows)
    input_combi.index = ["test_1", "test_2", "test_3", "test_4", "test_5"]
    test_set_df[fold] = input_combi

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
