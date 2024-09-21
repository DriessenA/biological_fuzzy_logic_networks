__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""
# modules defined in biofuzznet/
# Pylance throws a reportMissingImports but thos actually works.
from biological_fuzzy_logic_networks.utils import (
    MSE_entropy_loss,
    read_sif,
)  # weighted_and_mixed_loss
from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet
from biological_fuzzy_logic_networks.biofuzzdataset import BioFuzzDataset
from biological_fuzzy_logic_networks.mixed_gate import MixedGate
import warnings

# external python modules

import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class BioMixNet(BioFuzzNet):
    """
    This class implements a BioFuzzNet whose logical gates are a linear combination of AND and OR gates.
    AND and OR gates can still be implemented, but the default initialisation of a BioMixNet set all gates
        to MIXED status. Gates that should not be MIXED need to be manually changed to the correct node_type after
        initialisation of the network.
    """

    def __init__(
        self,
        nodes=None,
        edges=None,
        AND_param=0.0,
    ):
        """
        Initialises a BioMixNet.

        Args:
            nodes: list of nodes of the network
            edges: dict mapping tuples (upstream_node, downstream_node) to edge weight (which should be 1 or -1)
            AND_param: value at which to initialise the weight for an AND gates at the MIXED gates. It needs to be
                sigma^(-1) of the desired value since where sigma is the sigmoid function (see implementation
                of the MixedGate class). Default is 0 since sigma(0) = 0.5 (equal weight of the AND and OR gate)
                OR_param, the value at which the weight for an OR gates should be initialised at the MIXED gates,
                    is assumed to be 1 - AND_param
        """
        super().__init__(nodes, edges)

        # for node in self.nodes():
        #     if self.nodes()[node]["node_type"] in ["logic_gate_AND", "logic_gate_OR"]:
        #         self.nodes()[node]["node_type"] = "logic_gate_MIXED"
        #         self.nodes()[node]["gate"] = MixedGate(
        #             AND_param=AND_param,
        #             AND_function=self.integrate_AND,
        #             OR_function=self.integrate_OR,
        #         )

    def build_BioMixNet_from_file(filepath: str):
        """
        An alternate constructor to build the BioMixNet from the sif file instead of the lists of ndoes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return BioMixNet(nodes, edges)

    @property
    def mixed_gates(self):
        """Return the list of MIXED gates names in the network"""
        mixed_gates = [
            node
            for node, attributes in self.nodes(data=True)
            if attributes["node_type"] == "logic_gate_MIXED"
        ]
        return mixed_gates

    def add_fuzzy_node(
        self, node_name: str, type: str, AND_param=0.0
    ):  # torch.sigmoid(0) = 0.5
        """
        Add node to a BioFuzzNet
        Args:
            - node_name: name of the node which will be used to access it
            - type: type of the node. Should be one of BIO (biological), AND, OR, NOT (the last three being logical gate nodes)
            - AND_param: value at which to initialise the AND_param attribute of a MIXED gate
            - AND_function: value at which to initialise the OR_funciton attribute of a MIXED gate
            - OR_function: value at which to initialise the OR_funciton attribute of a MIXED gate
        """
        # Sanity check 1: the node type should belong to "BIO", "AND", "OR", "NOT" or "MIXED"
        types = ["BIO", "AND", "OR", "NOT", "MIXED"]
        if type not in types:
            ValueError(f"type should be in {types}")
        # Sanity check 2: node should not already exist
        if node_name in self.nodes():
            warnings.warn(f"Node {node_name} already exists, node was not added")
        # Add the nodes
        if type == "BIO":
            self.add_node(
                node_name, node_type="biological", output_state=None, ground_truth=None
            )
        if type == "AND":
            self.add_node(node_name, node_type="logic_gate_AND", output_state=None)
        if type == "NOT":
            self.add_node(node_name, node_type="logic_gate_NOT", output_state=None)
        if type == "OR":
            self.add_node(node_name, node_type="logic_gate_OR", output_state=None)
        if type == "MIXED":
            self.add_node(node_name, node_type="logic_gate_MIXED", output_state=None)
            self.nodes()[node_name]["gate"] = MixedGate(
                AND_param=AND_param,
                AND_function=self.integrate_AND,
                OR_function=self.integrate_OR,
            )

    def integrate_logical_node(self, node: str) -> torch.Tensor:
        """
        A wrapper around integrate_NOT, and the MixedGate layer to integrate the different logical nodes.
        Args:
            node: the name of the node representing the logical gate
        Returns:
            The state at the logical gate after integration

        """
        if self.nodes[node]["node_type"] == "logic_gate_AND":
            return self.integrate_AND(node)
        elif self.nodes[node]["node_type"] == "logic_gate_OR":
            return self.integrate_OR(node)
        elif self.nodes[node]["node_type"] == "logic_gate_NOT":
            return self.integrate_NOT(node)
        elif self.nodes[node]["node_type"] == "logic_gate_MIXED":
            return self.nodes[node]["gate"](node)
        else:
            raise NameError("This node is not a known logic gate.")

    def conduct_optimisation(
        self,
        input: dict,
        ground_truth: dict,
        test_input: dict,
        test_ground_truth: dict,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        mixed_gates_regularisation=1.0,
        optim_wrapper=torch.optim.Adam,
        loss_weights=None,
    ):
        """
        The main function of this class.
        Optimise the tranfer function parameters in a FIXED topology with VARIABLE logical gates, called MIXED gates.
        For the moment, the optimizer is ADAM.
        The loss function has been modified compared to the BioFuzzNet in order to incorporate a particular loss on the
            MIXED gates' parameters.
        Method overview:
            The graph states are updated by traversing the graph from root node to leaf node (forward pass).
            The transfer function parameters are then updated using backpropagation.
            The use of backpropagation forces the use of a sequential update scheme.

        Args:
            - input: dict of torch.Tensor mapping root nodes name to their input value
                (which is assumed to also be their ground truth value, otherwise those nodes will never be fitted correctly)
                It is assumed that every node in input is an input node that should be known to the model prior to simulation.
                Input nodes are then used as the start for the sequential update algorithm.
                input should usually contain the value at root nodes, but in the case where the graph contains a cycle,
                other nodes can be specified.
            - ground_truth: training dict of {node_name: torch.Tensor} mapping each observed biological node to its measured values
                Only  the nodes present in ground_truth will be used to compute the loss/
            - test_input: dict of torch.Tensor containing root node names mapped to the input validation data
            - test_ground_truth:  dict of torch.Tensor mapping node names to their value from the validation set
            - epochs: number of epochs for optimisation
            - batch_size: batch size for optimisation
            - learning_rate : learning rate for optimisation with ADAM
        - mixed_gates_regularisation: a float representing regularisation strength at the MIXED gates. Default 1.
            - optim_wrapper: a wrapper function for the optimiser. It should take as argument:
                - the parameters to optimise
                - the learning rate

        """
        torch.autograd.set_detect_anomaly(True)
        torch.set_default_tensor_type(torch.DoubleTensor)
        # Input nodes
        if len(self.root_nodes) == 0:
            input_nodes = [k for k in test_input.keys()]
        else:
            input_nodes = self.root_nodes

        # Instantiate the dataset
        dataset = BioFuzzDataset(input, ground_truth)

        # Instantiate the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Keep track of the parameters
        parameters = []
        for edge in self.transfer_edges:
            layer = self.edges()[edge]["layer"]
            parameters += [layer.n, layer.K]
        for gate in self.mixed_gates:
            gate = self.nodes[gate]["gate"]
            parameters += [
                gate.AND_param
            ]  # OR_param = 1 - AND_param so no need to add it

        # Instantiate the model
        self.initialise_random_truth_and_output(batch_size)

        # Set the parameters, leave possibility for other losses/solver
        if loss_weights is None:
            loss_weights = {n: 1 for n in self.biological_nodes}
        optim = optim_wrapper(parameters, learning_rate)

        # Train the model
        losses = pd.DataFrame(columns=["time", "loss", "phase"])
        for e in tqdm(range(epochs)):
            for X_batch, y_batch in dataloader:
                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                # Reinitialise the network
                self.initialise_random_truth_and_output(batch_size)
                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)
                # Simulate
                self.sequential_update(input_nodes)

                # Get the predictions
                predictions = self.output_states

                loss = MSE_entropy_loss(
                    predictions=predictions,
                    ground_truth=y_batch,
                    gates=[self.nodes[node]["gate"] for node in self.mixed_gates],
                    mixed_gates_regularisation=mixed_gates_regularisation,
                )
                # First reset then compute the gradients
                optim.zero_grad()
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(parameters, max_norm=5)
                # Update the parameters
                optim.step()
                # We save metrics with their time to be able to compare training vs test even though they are not logged with the same frequency
                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": loss.detach().item(),
                                "phase": "train",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

            with torch.no_grad():
                #   Instantiate the model
                self.initialise_random_truth_and_output(
                    len(test_ground_truth[self.root_nodes[0]])
                )
                self.set_network_ground_truth(ground_truth=test_ground_truth)
                # Simulation
                self.sequential_update(input_nodes)
                # Get the predictions
                predictions = self.output_states
                test_loss = MSE_entropy_loss(
                    predictions=predictions,
                    ground_truth=test_ground_truth,
                    gates=[self.nodes[node]["gate"] for node in self.mixed_gates],
                    mixed_gates_regularisation=mixed_gates_regularisation,
                )
                # No need to detach since there are no gradients
                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": test_loss.item(),
                                "phase": "test",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
        return losses
