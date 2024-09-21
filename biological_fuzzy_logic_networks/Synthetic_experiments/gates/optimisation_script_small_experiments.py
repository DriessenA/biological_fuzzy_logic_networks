# Imports
import torch
import argparse
import os
import pickle
import pandas as pd
import seaborn as sns

import biological_fuzzy_logic_networks.biomixnet as biomixnet
import biological_fuzzy_logic_networks.biofuzznet as biofuzznet
import biological_fuzzy_logic_networks.utils as utils

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Seed for reproducible simulations. The default here is 1 as data was simulated with default 2.",
)

parser.add_argument(
    "--inputnetwork",
    type=str,
    required=True,
    help="Relative path to the SIF file containing the network to optimise",
)

parser.add_argument(
    "--num_gates",
    type=str,
    required=True,
    help="Number of gates that were changed",
)


parser.add_argument(
    "--inputdata",
    type=str,
    required=True,
    help="Relative path to the folder containing the train and test data. Should end with a backslash",
)
parser.add_argument(
    "--outputfolder",
    type=str,
    required=True,
    help="Relative path to the folder where the optimisation output will be saved. Should end with a backslash. Folder will be created if it does not exist",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=80,
    help="Number of epochs for optimisation. Default = 80",
)
parser.add_argument(
    "--learningrate",
    type=float,
    default=0.05,
    help="Learning rate for optimisation. Default = 0.05.",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=300,
    help="Batch size for optimisation. Default = 300.",
)

parser.add_argument(
    "--fold",
    type=int,
    required=True,
    help="Which fold is used",
)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
num_gates = args.num_gates

# Load data
network_BFZ = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.inputnetwork)
network_BMN = biomixnet.BioMixNet.build_BioMixNet_from_file(args.inputnetwork)
node_names = pickle.load(open(f"{args.inputdata}node_names_{num_gates}.p", "rb"))
train_input = pickle.load(open(f"{args.inputdata}train_input_{num_gates}.p", "rb"))
test_input = pickle.load(open(f"{args.inputdata}test_input_{num_gates}.p", "rb"))
train_ground_truth = pickle.load(
    open(f"{args.inputdata}train_ground_truth_{num_gates}.p", "rb")
)
test_ground_truth = pickle.load(
    open(f"{args.inputdata}test_ground_truth_{num_gates}.p", "rb")
)

# See if output folder exists
# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")


# Set up the gates of the network
for gate in network_BMN.mixed_gates:
    if gate not in node_names:
        network_BMN.nodes()[gate]["node_type"] = network_BFZ.nodes()[gate]["node_type"]
        del network_BMN.nodes()[gate]["gate"]

# Optimise
losses = []
losses_i = network_BMN.conduct_optimisation(
    input=train_input,
    ground_truth=train_ground_truth,
    test_input=test_input,
    test_ground_truth=test_ground_truth,
    epochs=args.epochs,
    learning_rate=args.learningrate,
    batch_size=args.batchsize,
)
losses.append(losses_i)


# Concatenate the losses of the iterative runs
all_losses = pd.concat(losses)
all_losses.reset_index(inplace=True)
all_losses

# Plot the losses to check optimisation happened correctly
plot_loss = sns.lineplot(data=all_losses, x="time", y="loss", hue="phase")
plot_fig = plot_loss.get_figure()
plot_fig.savefig(f"{args.outputfolder}/total_loss_{args.fold}.png")

# Save the data we will need later
parameters_opt, n_opt, K_opt = utils.obtain_params(network_BMN)
pickle.dump(
    parameters_opt, open(f"{args.outputfolder}/parameters_opt_{args.fold}.p", "wb")
)
pickle.dump(network_BMN, open(f"{args.outputfolder}/model_{args.fold}.p", "wb"))
pickle.dump(all_losses, open(f"{args.outputfolder}/losses_{args.fold}.p", "wb"))

with open(f"{args.outputfolder}/hyperparameters_{args.fold}.txt", "w") as f:
    f.write(
        f"seed: {args.seed},  input network: {args.inputnetwork}, epochs: {args.epochs}, batchsize: {args.batchsize}, rounds: {args.rounds}, learningrate: {args.learningrate}, gates: {node_names}, fold: {args.fold}"
    )
    f.close()
