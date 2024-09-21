__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

# modules defined in biofuzznet/
# Pylance throws a reportMissingImports but thos actually works.

import copy
import warnings
from datetime import datetime
from typing import Optional

import pandas as pd
import torch

# external python modules
from networkx.classes.digraph import DiGraph
from tqdm import tqdm

from biological_fuzzy_logic_networks.Hill_function import HillTransferFunction
from biological_fuzzy_logic_networks.biofuzzdataset import BioFuzzDataset
from biological_fuzzy_logic_networks.utils import (
    has_cycle,
    read_sif,
    MSE_loss,
)  # , weighted_loss


class BioFuzzNet(DiGraph):
    """This class represents a BioFuzzNet, that is a Boolean biological network
    on which fuzzy logic operations can be implemented."""

    def __init__(self, nodes=None, edges=None, n=2, K=0.5):
        """
        Initialise a BioFuzzNet.
        Logical AND gates should be defined in the nodes and edges list, by having a node name
        containing _and_. Otherwise, for all nodes having more than 1 incoming edges, those edges
        are assumed to be linked together by an OR gate

        Args:
            - nodes: list of nodes of the network
            - edges: dict mapping tuple (upstream_edge, downstream_edge) to edge weight
             (which should be 1 or -1)

        Default initialises an empty BioFuzzNet

        """
        super().__init__()

        if nodes is not None and edges is not None:
            # Start by building the graph
            for node in nodes:
                # Then we have to check manually is some of the nodes are AND gates
                if (
                    "_and_" in node
                ):  # Different convention than CellNOpt for more readability
                    # This node is an AND gate
                    self.add_fuzzy_node(node, "AND")
                elif (
                    "_or_" in node
                ):  # Different convention than CellNOpt for more readability
                    # This node is an OR gate
                    self.add_fuzzy_node(node, "OR")
                else:
                    self.add_fuzzy_node(node, "BIO")
            # At this point all biological nodes have been added
            # Add edges, adding NOT gates when necessary
            not_count = 1
            for edge, edge_weight in edges.items():
                if int(edge_weight) == -1:
                    self.add_negative_edge(edge, not_count)
                else:
                    if self.nodes()[edge[0]]["node_type"] == "biological":
                        self.add_transfer_edge(edge[0], edge[1], n=n, K=K)
                    else:
                        self.add_simple_edge(edge[0], edge[1])

            #  Deal with the OR gates
            self.resolve_OR_gates()

    # Property methods

    # networkx subgraph_view and subgraph_functions are not used to filter nodes and edges
    # as they call BioFuzzNet.biofuzznet.__init__(). Since the constructor of a BioFuzzNet
    # takes additional arguments, this throws an error.

    @property
    def biological_nodes(self):
        """Return a list containing the names of the biological nodes."""
        biological_nodes = [
            node
            for node, attributes in self.nodes(data=True)
            if attributes["node_type"] == "biological"
        ]
        return biological_nodes

    @property
    def fuzzy_nodes(self):
        """Return a list containing the names of the fuzzy nodes."""
        fuzzy_nodes = [
            node
            for node, attributes in self.nodes(data=True)
            if attributes["node_type"] in ["logic_gate_AND", "logic_gate_OR"]
        ]
        return fuzzy_nodes

    @property
    def root_nodes(self):
        """
        Return a list containing the names of the root nodes.
        Throw a warning if the network does not have any root node.
        """
        root_nodes = [
            node for node in self.nodes() if len(list(self.predecessors(node))) == 0
        ]
        if len(root_nodes) == 0:
            warnings.warn(
                "No root nodes in the network, most probably due to a loop. Be sure to specify input nodes for optimisation."
            )
        return root_nodes

    @property
    def leaf_nodes(self):
        """
        Return a list containing the names of the leaf nodes.
        Throw a warning if the network does not have any leaf node.
        """
        leaf_nodes = [
            node for node in self.nodes() if len(list(self.successors(node))) == 0
        ]
        if len(leaf_nodes) == 0:
            warnings.warn("No leaf nodes in the network, most probably due to a loop.")
        return leaf_nodes

    @property
    def output_states(self):
        """
        Return a dict mapping biological node names to their output state.
        """
        output_states = {
            node: self.nodes()[node]["output_state"] for node in self.biological_nodes
        }
        return output_states

    @property
    def transfer_edges(self):
        """
        Return a list of all edges sporting a transfer function.
        """
        transfer_edges = [
            (upstream_node, downstream_node)
            for upstream_node, downstream_node, attributes in self.edges(data=True)
            if attributes["edge_type"] == "transfer_function"
        ]
        return transfer_edges

    # Building methods

    """ The goal of those methods is to make sure the edges and nodes are added with the correct attributes"""

    def add_simple_edge(self, upstream_node: str, downstream_node: str) -> None:
        """
        Add simple directed edge (without a transfer function) to the network
        Args:
            - upstream_node: name of the source node
            - downstream_node: name of the target node
        """
        # Sanity check 1: Throw error if nodes don't exist
        if upstream_node not in self.nodes():
            ValueError(
                f"Node {upstream_node} is not in the graph. Add it using the corresponding function."
            )
        if downstream_node not in self.nodes():
            ValueError(
                f"Node {downstream_node} is not in the graph. Add it using the corresponding function."
            )
        # The weight is needed to keep the directedness of the networkX graph and use the corresponding functions
        self.add_edge(upstream_node, downstream_node, edge_type="simple", weight=1)

    def add_transfer_edge(
        self,
        upstream_node: str,
        downstream_node: str,
        K=0.5,
        n=2,
    ) -> None:
        """
        Add transfer directed edge (with a Hill transfer function) to the network
        Args:
            - upstream_node: name of the source node
            - downstream_node: name of the target node

        POSSIBLE UPGRADE:
            - Be able to specify the layer we want to have as a transfer function
        """
        # Sanity check 1: Throw error if nodes don't exist
        if upstream_node not in self.nodes():
            ValueError(
                f"Node {upstream_node} is not in the graph. Add it using the corresponding function."
            )
        if downstream_node not in self.nodes():
            ValueError(
                f"Node {downstream_node} is not in the graph. Add it using the corresponding function."
            )

        # Sanity check 2: A transfer edge can only occur after a biological node
        if self.nodes()[upstream_node]["node_type"] != "biological":
            ValueError(
                f"Only biological nodes can be directly upstream of a transfer function. Check node {upstream_node}."
            )

        self.add_edge(
            upstream_node,
            downstream_node,
            edge_type="transfer_function",
            layer=HillTransferFunction(K=K, n=n),
            weight=1,
        )

    def add_fuzzy_node(self, node_name: str, type: str) -> None:
        """
        Add node to a BioFuzzNet
        Args:
            - node_name: name of the node which will be used to access it
            - type: type of the node. Should be one of BIO (biological), AND, OR, NOT (the last three being logical gate nodes)
        """
        # Sanity check 1: the node type should belong to "BIO", "AND", "OR", "NOT"
        types = ["BIO", "AND", "OR", "NOT"]
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

    def add_negative_edge(self, edge: tuple, not_counter: int) -> int:
        """
        Add a negative edge to a BioFuzzNet.
        Create the NOT gate and add it to the network.
        Modify edges accordingly: the edge preceding the NOT gate is a transfer edge if it starts at a biological node, a simple edge otherwise.

        Args:
            edge: the negative edge to add
            not_counter: the number to assign to the node edge
        Returns:
            the incremented not_counter
        """
        not_node = f"not{not_counter}"
        while not_node in self.nodes():
            not_counter += 1
            not_node = f"not{not_counter}"
        # Start by adding the node as we cannot add edges if the nodes do not exist
        self.add_fuzzy_node(not_node, "NOT")
        if self.nodes()[edge[0]]["node_type"] == "biological":
            self.add_transfer_edge(edge[0], not_node)
        else:
            self.add_simple_edge(edge[0], not_node)
        self.add_simple_edge(not_node, edge[1])
        return not_counter

    def resolve_OR_gates(self) -> None:
        """
        For use on a BioFUZZNet that is not yet fully implemented.
        Make sure that the structure of the DiGraph corresponds to the structure
        enforced for the BioFuzzNet, by adding the OR gates where necessary.
        We assume that all AND gates have been resolved already, hence any nodes that
        combines two inputs or more and is not registered as an AND gate has to
        integrate those inputs through an OR gate.
        """
        or_counts = 1
        for b_node in self.biological_nodes:
            predecessors = [pred for pred in self.predecessors(b_node)]
            if len(predecessors) > 1:
                curr_parent = predecessors.pop(0)
                while predecessors:
                    # If there is more than 2 input at an OR gate, I distribute them in 2 inputs OR gates
                    # ie (A OR B OR C OR D) becomes (((A OR B) OR C) OR D)
                    second_parent = predecessors.pop(0)
                    or_node = f"or{or_counts}"
                    or_counts += 1
                    self.add_fuzzy_node(or_node, "OR")
                    for pred in [curr_parent, second_parent]:
                        if self.edges()[(pred, b_node)]["edge_type"] == "simple":
                            self.remove_edge(pred, b_node)
                            self.add_simple_edge(pred, or_node)
                            self.add_simple_edge(
                                or_node, b_node
                            )  # Not the most efficient when I have more than 2 inputs as I create edges that I will later destroy
                        else:
                            self.remove_edge(pred, b_node)
                            self.add_transfer_edge(pred, or_node)
                            self.add_simple_edge(or_node, b_node)
                        curr_parent = or_node

                self.add_simple_edge(or_node, b_node)
        # If I have several inputs to an AND gate then there's a mistake
        for node, attributes in self.nodes(data=True):
            predecessors = list(self.predecessors(node))
            if attributes["node_type"] == "logic_gate_AND" and len(predecessors) > 2:
                raise ValueError(
                    f"AND gate {node} has too many inputs, please explicitly indicate AND and OR gates in the SIF file. "
                )

    @classmethod
    def build_BioFuzzNet_from_file(cls, filepath: str, n=2, K=0.5):
        """
        An alternate constructor to build the BioFuzzNet from the sif file instead of the lists of nodes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return BioFuzzNet(nodes, edges, n=n, K=K)

    # Setter Methods
    def initialise_random_truth_and_output(self, batch_size):
        """
        Initialises the network so that the output_state and ground_truth are set to random tensors.
        Args:
            - batch_size: size of the tensor. All tensors will have the same size.
        NB: This is useful because output_state and ground_truth are set to None when adding nodes using self.add_fuzzy_node()
            and having None values creates unwanted behavior when using mathematical operations (NaN propagates to non-NaN tensors)
        """
        for node_name in self.nodes():
            node = self.nodes()[node_name]
            if node["node_type"] == "biological":
                node["ground_truth"] = torch.rand(batch_size)
                node["output_state"] = torch.rand(batch_size)
            else:
                node["output_state"] = torch.rand(batch_size)

    def set_network_ground_truth(self, ground_truth):
        """
        Set the ground_truth of each biological node. Throws a warning for each biological node
        in the BioFuzzNet that is not observed
        Args:
            - ground_truth: a dict mapping the name of each biological node to a tensor representing its ground_truth.
        NB: No ground truth value is set for non-measured nodes, the loss function should thus be consequentially chosen
        """
        # First check that all root nodes at least have an input
        missing_inputs = []
        for node in self.root_nodes:
            if node not in ground_truth.keys():
                missing_inputs.append(node)
        if len(missing_inputs) > 0:
            raise ValueError(f"Missing input values for root nodes {missing_inputs}")

        for node_name in self.biological_nodes:
            parents = [p for p in self.predecessors(node_name)]
            if node_name in ground_truth.keys():
                node = self.nodes()[node_name]
                if (
                    len(parents) > 0
                ):  # If the node has a parent (ie is not an input node for which we for sure have the ground truth as prediction)
                    node["ground_truth"] = ground_truth[node_name]
                else:
                    node["ground_truth"] = ground_truth[node_name]
                    node["output_state"] = ground_truth[
                        node_name
                    ]  # A root node does not need to be predicted
            # else:
            #     warnings.warn(
            #         f"Node {node_name} is not present in ground truth. Its ground_truth value is not set."
            #     )

    # Update methods
    def propagate_along_edge(self, edge: tuple) -> torch.Tensor:
        """
        Transmits node state along an edge.
        If an edge is simple: then it returns the state at the upstream node. No computation occurs in this case.
        If an edge sports a transfer function: then it computes the transfer function and returns the transformed state.

        Args:
            edge: The edge along which to propagate the state
        Returns:
            The new state at the target node of the edge
        """
        if edge not in self.edges():
            raise NameError(f"The input edge {edge} does not exist.")
            assert False
        if self.edges()[edge]["edge_type"] == "simple":
            state_to_propagate = self.nodes[edge[0]]["output_state"]
            return state_to_propagate
        if self.edges()[edge]["edge_type"] == "transfer_function":
            # The preceding state has to go through the Hill layer
            state_to_propagate = self.edges()[edge]["layer"](
                self.nodes[edge[0]]["output_state"]
            )
            return state_to_propagate
        else:
            NameError("The node type is incorrect")
            assert False

    def integrate_NOT(self, node: str) -> torch.Tensor:
        """
        Computes the NOT operation at a NOT gate

        Args:
            node: the name of the node representing the NOT gate
        Returns:
            The output state at the NOT gate after computation
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 1:
            raise AssertionError("This NOT gate has more than one predecessor")
        if len(upstream_edges) == 0:
            raise AssertionError("This NOT gate has no predecessor")
        else:
            state_to_integrate = self.propagate_along_edge(upstream_edges[0])
            ones = torch.ones(state_to_integrate.size())
            result = ones - state_to_integrate
            # TODO check if this is reasonable
            zeroes = torch.isclose(result, torch.zeros(len(result)))
            result[zeroes] = 0
            ones = torch.isclose(result, torch.ones(len(result)))
            result[ones] = 1
            # We work with tensors
            return result

    def integrate_AND(self, node: str) -> torch.Tensor:
        """
        Integrate the state values from all incoming nodes at an AND gate.
        Cannot support more than two input gates.

        Args:
            node: the name of the node representing the AND gate
        Returns:
            The output state at the AND gate after integration
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 2:
            raise AssertionError(
                f"The AND gate {node} has more than two incoming edges."
            )
        states_to_integrate = [
            self.propagate_along_edge(edge) for edge in upstream_edges
        ]
        # Multiply all the tensors
        result = states_to_integrate[0] * states_to_integrate[1]
        # TODO check if this is reasonable
        zeroes = torch.isclose(result, torch.zeros(len(result)))
        result[zeroes] = 0
        ones = torch.isclose(result, torch.ones(len(result)))
        result[ones] = 1
        return result

    def integrate_OR(self, node: str) -> torch.Tensor:
        """
        Integrate the state values from all incoming nodes at an OR gate.
        Cannot support more than two input gates.

        Args:
            node: the name of the node representing the OR gate
        Returns:
            The state at the OR gate after integration
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 2:
            raise AssertionError(
                f"The OR gate {node} has more than two incoming edges."
            )
        states_to_integrate = [
            self.propagate_along_edge(edge) for edge in upstream_edges
        ]

        # Multiply all the tensors
        result = (
            states_to_integrate[0]
            + states_to_integrate[1]
            - states_to_integrate[0] * states_to_integrate[1]
        )
        # TODO check if this is reasonable
        zeroes = torch.isclose(result, torch.zeros(len(result)))
        result[zeroes] = 0
        ones = torch.isclose(result, torch.ones(len(result)))
        result[ones] = 1
        return result

    def integrate_logical_node(self, node: str) -> torch.Tensor:
        """
        A wrapper around integrate_NOT, integrate_OR and integrate_AND to integrate the values
        at any logical node independently of the gate.

        Args:
            - node: the name of the node representing the logical gate
        Returns:
            - The state at the logical gate after integration

        """
        if self.nodes[node]["node_type"] == "logic_gate_AND":
            return self.integrate_AND(node)
        if self.nodes[node]["node_type"] == "logic_gate_OR":
            return self.integrate_OR(node)
        if self.nodes[node]["node_type"] == "logic_gate_NOT":
            return self.integrate_NOT(node)
        else:
            raise NameError("This node is not a known logic gate.")

    def update_biological_node(self, node: str) -> torch.Tensor:
        """
        Returns the updated output state of a node when propagating  through the graph.
        Args:
            - node: name of the biological node to update
        Return:
            - a torch.Tensor representing the updated value of the node
        """
        parent_node = [p for p in self.predecessors(node)]
        if len(parent_node) > 1:
            raise AssertionError("This biological node has more than one incoming edge")
        elif len(parent_node) == 1:
            # The state of a root node stays the same
            return self.propagate_along_edge((parent_node[0], node))
        else:  # For a root edge
            return self.nodes()[node]["ground_truth"]

    def update_fuzzy_node(self, node: str) -> None:
        """
        A wrapper to call the correct updating function depending on the type of the node.
        Args:
            - node: name of the node to update
        """
        node_type = self.nodes()[node]["node_type"]
        if node_type == "biological":
            self.nodes()[node]["output_state"] = self.update_biological_node(node)
        else:
            self.nodes()[node]["output_state"] = self.integrate_logical_node(node)

    def update_one_timestep_cyclic_network(
        self, input_nodes, loop_status, convergence_check=False
    ) -> Optional[dict]:
        """
        Does the sequential update of a directed cyclic graph over one timestep: ie updates each node in the network only once.
        Args:
            - input_nodes: the node to start updating from, ie those for which we give the ground truth as input to the model
            - loop_status: the value returned by utils.has_cycle(self) which is a tuple (bool, list) where bool is True if the
            graph has a directed cycle, and the list is the list of all directed cycles in the graph
            - convergence_check: default False. In case one wants to check convergence of the simulation
                 for a graph with a loop, this Boolean should be set to True, and output state of the one-step simulation will be saved and returned. This has however not been
                 optimised for time and memory usage. Use with caution.
        """
        if convergence_check:
            warnings.warn(
                "convergence_check has been set to True. All simulation states will be saved and returned. This has not been optimised for memory usage and is implemented in a naive manner. Proceed with caution."
            )

        current_nodes = copy.deepcopy(input_nodes)
        non_updated_nodes = [n for n in self.nodes()]
        while len(non_updated_nodes) > 0:
            # curr_nodes is a queue, hence FIFO (first in first out)
            # when popping the first item, we obtain the one that has been in the queue the longest
            curr_node = current_nodes.pop(0)
            # If the node has not yet been updated
            if curr_node in non_updated_nodes:
                can_update = False
                non_updated_parents = [
                    p for p in self.predecessors(curr_node) if p in non_updated_nodes
                ]
                # Check if parents are updated
                if len(non_updated_parents) > 0:
                    for p in non_updated_parents:
                        # Check if there is a loop to which both the parent and the current node belong
                        for cycle in loop_status[1]:
                            if curr_node in cycle and p in cycle:
                                # Then we will need to update curr_node without updating its parent
                                non_updated_parents.remove(p)
                                break
                    # Now non_updated_parents only contains parents that are not part of a loop to which curr_node belongs
                    if len(non_updated_parents) > 0:
                        can_update = False
                        for p in non_updated_parents:
                            current_nodes.append(p)
                    else:
                        can_update = True
                    # The parents that were removed will be updated later as they are still part of non_updated nodes
                else:  # If all node parents are updated then no problem
                    can_update = True
                if not can_update:
                    # Then we reappend the current visited node
                    current_nodes.append(curr_node)
                else:  # Here we can update
                    print(curr_node)
                    self.update_fuzzy_node(curr_node)
                    non_updated_nodes.remove(curr_node)
                    cont = True
                    while cont:
                        try:
                            current_nodes.remove(curr_node)
                        except ValueError:
                            cont = False
                    child_nodes = [c for c in self.successors(curr_node)]
                    for c in child_nodes:
                        if c in non_updated_nodes:
                            current_nodes.append(c)
        if convergence_check:
            return self.output_states  # For checking convergence
        else:
            return None

    def sequential_update(self, input_nodes, convergence_check=False) -> Optional[dict]:
        """
        Update the graph by propagating the signal from root node (or given input node)
        to leaf node. This update is sequential according to Boolean networks terminology.

        Method overview:
            The graph is traversed from root node to leaf node.
            The list of the nodes to be updated is implemented as a queue in a First In First Out (FIFO)
                 in order to update parents before their children.

        Args:
            - input_nodes: Nodes for which the ground truth is known and used as input for simulation (usually root nodes)
            - convergence_check: default False. In case one wants to check convergence of the simulation
                 for a graph with a loop, this Boolean should be set to True, and output states of the model
                 over the course of the simulation will be saved and returned. This has however not been
                 optimised for memory usage. Use with caution.
        """
        if convergence_check:
            warnings.warn(
                "convergence_check has been set to True. All simulation states will be saved and returned. This has not been optimised for memory usage and is implemented in a naive manner. Proceed with caution."
            )
        states = {}
        loop_status = has_cycle(self)
        if not loop_status[0]:
            current_nodes = copy.deepcopy(input_nodes)
            non_updated_nodes = [n for n in self.nodes()]
            safeguard = 0
            node_number = len([n for n in self.nodes()])
            while len(non_updated_nodes) > 0:
                safeguard += 1
                if safeguard > 10 * node_number:
                    print(
                        "Safeguard activated at 10*total number of nodes repetitions. Check if your network has loops. If node augment the safeguard."
                    )
                    break

                # curr_nodes is FIFO
                curr_node = current_nodes.pop(0)
                # If the node has not yet been updated
                if curr_node in non_updated_nodes:
                    parents = [pred for pred in self.predecessors(curr_node)]
                    non_updated_parents = [p for p in parents if p in non_updated_nodes]
                    # If one parent is not updated yet, then we cannot update
                    if len(non_updated_parents) > 0:
                        for p in non_updated_parents:
                            # curr_nodes is FIFO: we first append the parents then the child
                            current_nodes.append(p)
                        current_nodes.append(curr_node)
                    # If all parents are updated, then we update
                    else:
                        self.update_fuzzy_node(curr_node)
                        non_updated_nodes.remove(curr_node)
                        cont = True
                        while cont:
                            try:
                                current_nodes.remove(curr_node)
                            except ValueError:
                                cont = False
                        child_nodes = [c for c in self.successors(curr_node)]
                        for c in child_nodes:
                            if c in non_updated_nodes:
                                current_nodes.append(c)
        else:
            # The time of the simulation is 2 times the size of the biggest cycle
            length = 3 * max([len(cycle) for cycle in has_cycle(self)[1]])
            # We simulate length times then output the mean of the last length simulations
            # CHANGED length to int(length/2)
            states[0] = self.output_states
            for i in range(1, int(length)):
                states[i] = self.update_one_timestep_cyclic_network(
                    input_nodes, loop_status, convergence_check
                )
            last_states = {}
            for i in range(int(length)):
                states[length + i] = self.update_one_timestep_cyclic_network(
                    input_nodes, loop_status, convergence_check
                )
                last_states[i] = {
                    n: self.nodes()[n]["output_state"] for n in self.nodes()
                }
            # Set the output to the mean of the last steps

            for n in self.nodes():
                output_tensor = last_states[0][n]
                for i in range(int(length) - 1):
                    output_tensor = output_tensor + last_states[i + 1][n]
                output_tensor = output_tensor / length
                self.nodes()[n]["output_state"] = output_tensor
        if convergence_check:
            return states  # For checking convergence
        else:
            return None

    # Optimisation methods

    def conduct_optimisation(
        self,
        input: dict,
        ground_truth: dict,
        test_input: dict,
        test_ground_truth: dict,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optim_wrapper=torch.optim.Adam,
    ):
        """
        The main function of this class.
        Optimise the tranfer function parameters in a FIXED topology with FIXED input gates.
        For the moment, the optimizer is ADAM and the loss function is the MSELoss over all observed nodes (see utils.MSE_Loss)
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
            - optim_wrapper: a wrapper function for the optimiser. It should take as argument:
                - the parameters to optimise
                - the learning rate

        POSSIBLE UPDATES:
            - Allow tuning between AND and OR gates using backpropagation
        """

        torch.autograd.set_detect_anomaly(True)
        torch.set_default_tensor_type(torch.DoubleTensor)
        # Input nodes
        if len(self.root_nodes) == 0:
            input_nodes = [k for k in test_input.keys()]
            print(f"There were no root nodes, {input_nodes} were used as input")
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

        optim = optim_wrapper(parameters, learning_rate)

        # Train the model
        losses = pd.DataFrame(columns=["time", "loss", "phase"])

        epoch_pbar = tqdm(range(epochs), desc="Valid loss=?.??e??")
        for e in epoch_pbar:
            # Instantiate the model
            self.initialise_random_truth_and_output(batch_size)

            for X_batch, y_batch in dataloader:
                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                # Reinitialise the network at the right size
                batch_keys = list(X_batch.keys())
                self.initialise_random_truth_and_output(len(X_batch[batch_keys.pop()]))
                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)
                # Simulate
                self.sequential_update(input_nodes)

                # Get the predictions
                predictions = self.output_states

                loss = MSE_loss(predictions=predictions, ground_truth=y_batch)

                # First reset then compute the gradients
                optim.zero_grad()

                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_value_(parameters, clip_value=5)
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
                # Instantiate the model
                self.initialise_random_truth_and_output(
                    len(test_ground_truth[input_nodes[0]])
                )
                self.set_network_ground_truth(ground_truth=test_ground_truth)
                # Simulation
                self.sequential_update(input_nodes)
                # Get the predictions
                predictions = self.output_states
                test_loss = MSE_loss(
                    predictions=predictions, ground_truth=test_ground_truth
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
            epoch_pbar.set_description(f"Valid loss:{test_loss.detach().item():.2e}")

        return losses
