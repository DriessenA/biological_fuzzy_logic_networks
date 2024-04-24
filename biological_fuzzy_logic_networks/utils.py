__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
import torch
import networkx as nx
from typing import Tuple
from math import exp
import matplotlib.pyplot as plt
import numpy as np


def read_sif(filepath: str) -> Tuple[list, dict]:
    """
    Read a SIF file and returns the list of node names and a dictionnary mapping edges to their weight.
    Args:
        - filepath: path to the SIF file
    Edges are assumed to be described in the order "source weight target"
    File is assumed to be space or tab separated
    """
    node_names = []
    edges = {}
    sif_file = open(filepath, "r")
    line = sif_file.readline()
    while line:
        line.strip()
        # print(line)
        node_1, edge_weight, node_2 = line.split()
        edge = (node_1, node_2)
        if edge not in edges:
            edges[edge] = edge_weight
        if node_1 not in node_names:
            node_names.append(node_1)
        if node_2 not in node_names:
            node_names.append(node_2)
        line = sif_file.readline()
    sif_file.close()
    return (node_names, edges)


def change_SIF_convention(filepath_in: str, filepath_out: str) -> None:
    """
    For a SIF file with convention "source target weight", return the corresponding SIF file with convention "source weight target" readable by read_SIF.
    Args:
        - filepath_in: path to the input file
        - filepath_out: path at which to save the output file
    """
    file_in = open(filepath_in, "r")
    file_out = open(filepath_out, "w")
    line = file_in.readline()
    while line:
        line.strip()
        # print(line)
        node_1, node_2, edge_weight = line.split()
        file_out.writelines("\t".join([node_1, edge_weight, node_2]))
        file_out.writelines("\n")
        line = file_in.readline()
    file_in.close()
    file_out.close()


def has_cycle(G: nx.DiGraph) -> Tuple[bool, list]:
    """
    For a BioFuzzNet, returns a Tuple(boolean, list), where the boolean indicates whether this BioFuzzNet
    contains a cycle, and the list is the list of cyles in the network, where each cycles is represented by
    the list of its nodes composing each cycles. See the documentation for
    networkX.recursive_simple_cycles.
    WARNING: This function uses NetworkX's recursive_simple_cycles function which uses up a lot of RAM.
     The simple_cycles function was not used because it failed to detect cycles on a toy network
    Args:
        - G: nx.DiGraph, or more specifically a BioFuzzNet
    Returns:
        - Tuple(bool, list): the boolean indicates whether this BioFuzzNet
    contains a cycle, and the list is the list of nodes composing each cycles
    """
    cycle_list = list(nx.recursive_simple_cycles(G))
    has_a_cycle = len(cycle_list) > 0
    return (has_a_cycle, cycle_list)


def dictionnary_to_tensor(output_dict) -> torch.Tensor:
    """
    Tranforms a dictionnary representing the output or ground truth of a BioFuzzNet
    into a tensor matrix of shape number_of_nodes * number_of_cells.
    Args:
        output_dict: dict mapping nodes of a BioFuzzNet to a tensor of values
    Returns:
        a tensor matrix of shape number_of_nodes * number_of_cells
    """
    keys = list(output_dict.keys())
    node_number = len(keys)  # Features
    k = keys.pop()
    cell_number = len(output_dict[k])  # Samples
    # Get list of tensors to concatenate
    to_concat = list(output_dict.values())
    matrix = torch.cat(to_concat)
    matrix = matrix.reshape((node_number, cell_number))
    return matrix


def MSE_loss(predictions: dict, ground_truth: dict) -> torch.Tensor:
    """
    Compute the MSE loss over all nodes of the network
    Args :
        - predictions: dict mapping each node to its predicted value
        - ground_truth: dict mapping each node to its ground_truth.
            Unobserved nodes should not be present in ground truth.
    Returns:
        - a torch.tensor containing the minibatch MSE loss over all observed nodes in ground_truth
    """
    # Remove unobserved nodes from the prediction
    # And reorder the predictions to be in the same order as the ground truth
    predictions = {key: predictions[key] for key in ground_truth.keys()}

    # Get the matrices
    predictions = dictionnary_to_tensor(predictions)
    ground_truth = dictionnary_to_tensor(ground_truth)
    # Compute the squared loss without any reduction
    mse_loss = torch.nn.MSELoss(reduction="none")
    squared_loss = mse_loss(predictions, ground_truth)

    # Then I can average however I want
    # I will then average over the network nodes
    loss = torch.mean(squared_loss, 0)

    # Then I average over the batch
    loss = torch.mean(loss)
    return loss


def MSE_entropy_loss(
    predictions, ground_truth, mixed_gates_regularisation, gates
) -> torch.Tensor:
    """
    Compute a MSE loss mixed with a separate loss for regularising the MIXED gates in BioMixNets.
    Args:
        - predictions: dict mapping each node to its predicted value
        - ground_truth: dict mapping each node to its ground_truth. Unobserved nodes should not be present in ground truth.
        - mixed_gates_regularisation: parameters for the regularisation of the mixed gates. If it has value p_reg, for each mixed gate,
            we add the value p_reg*AND_param*(1-AND_param)
        - gates: list of mixed gates in the network
    Returns:
        - a torch.tensor containing the minibatch MSE entropy loss over all observed nodes in ground_truth
    """
    mse_loss = MSE_loss(predictions=predictions, ground_truth=ground_truth)
    regularisation_loss = 0
    for mixed_gate in gates:
        regularisation_loss = regularisation_loss + (
            torch.sigmoid(mixed_gate.AND_param)
            * (1 - torch.sigmoid(mixed_gate.AND_param))
        )

    loss = mse_loss + mixed_gates_regularisation * regularisation_loss
    return loss


def obtain_params(G) -> Tuple[dict, list, list]:
    """
    Return a tuple of the list of values taken by parameters n and K
        of a HillTransferFunction from a BioFuzzNet.

    Args:
        A BioFuzzNet

    Return:
        Tuple[dictionnary mapping transfer edges to their parameter values,
            list of values of n,
            list of values of K]
    """
    param_dict = {
        e: [p.item() for p in G.edges()[e]["layer"].parameters()]
        for e in G.transfer_edges
    }
    n = []
    K = []
    for edge, params in param_dict.items():
        ni = exp(params[0])
        Ki = exp(params[1])
        n.append(ni)
        K.append(Ki)
    return (param_dict, n, K)


def param_dict_to_lists(param_dict) -> Tuple[list, list]:
    """
    Separate a dictionnary mapping transfer edges with HillTransferFunction
        to the values of the parameters of the HillTransferFunction into two
        lists of parameter values

    Args:
        param_dict: a dictionnary mapping transfer edges with HillTransferFunction
            to the values of the parameters of the HillTransferFunction
    """
    n = []
    K = []
    for edge, params in param_dict.items():
        ni = torch.exp(params[0]).item()
        Ki = torch.exp(params[1]).item()
        n.append(ni)
        K.append(Ki)
    return (n, K)


def compute_MSE(list_1: list, list_2: list):
    """
    Compute the MSE between two same-length list of parameters.
    Squared error is computed between list_1[i] and list_2[i].

    Args:
        - list_1: a list of values
        - list_2: a list of values
    Return:
        Mean Squared Error between the elements of those 2 lists
    """

    squared_error = [(list_1[i] - list_2[i]) ** 2 for i in range(len(list_1))]
    return sum(squared_error) / len(squared_error)


def compute_RMSE_outputs(model, ground_truth) -> dict:
    """
    Compute the RMSE between the model's output states and the observed ground truth

    Args:
        - model: a BioFuzzNet
        - ground_truth: a dicitonnary mapping node names to observed values (represented by torch.Tensors)
    Returns:
        - a dictionnary of the Root Mean Squared Error between the the model's output
        state and the ground truth for each node. Keys are the node names, values are the RMSE
    """
    rmse = {}
    for node in ground_truth.keys():
        rmse[node] = torch.sqrt(
            torch.sum((model.output_states[node] - ground_truth[node]) ** 2)
            / len(model.output_states[node])
        ).item()
    return rmse


def plot_edge_function(transfer_edge, low=0, high=1):
    """
    Plot the transfer function at a transfer edge


        Args:
            - transfer_edge: the transfer edge of a Biofuzznet
            - low: lower bound of the interval to plot
            - high: upper bound of the interval to plot
        Returns:
            None
    """
    hill = transfer_edge["layer"]
    xaxis = torch.tensor(np.linspace(low, high, 1000))
    hill.forward(xaxis)
    plt.plot(xaxis, hill.output_value.detach().numpy())
    plt.show()
