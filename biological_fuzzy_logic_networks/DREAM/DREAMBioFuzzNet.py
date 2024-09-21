import copy
import warnings
from datetime import datetime
from typing import Optional

import networkx as nx
import pandas as pd
import torch as torch
from tqdm.autonotebook import tqdm
from torch.distributions.uniform import Uniform

from biological_fuzzy_logic_networks.DREAM.DREAMdataset import DREAMBioFuzzDataset
from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet
from biological_fuzzy_logic_networks.utils import MSE_loss, read_sif, MSE_entropy_loss
from biological_fuzzy_logic_networks.utils import has_cycle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DREAMMixIn:
    # Setter Methods
    def initialise_random_truth_and_output(
        self,
        batch_size,
        distribution: torch.distributions.distribution.Distribution = Uniform(
            torch.tensor([0.0]), torch.tensor([1.0])
        ),
    ):
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
                sample1 = distribution.sample(torch.Size([batch_size]))
                sample2 = distribution.sample(torch.Size([batch_size]))

                sample1[(sample1 < 0)] = 0
                sample1[(sample1 > 1)] = 1

                sample2[(sample2 < 0)] = 0
                sample2[(sample2 > 1)] = 1

                node["ground_truth"] = sample1.reshape(
                    [
                        batch_size,
                    ]
                )
                node["output_state"] = sample2.reshape(
                    [
                        batch_size,
                    ]
                )
            else:
                sample = distribution.sample(torch.Size([batch_size]))

                sample[(sample < 0)] = 0
                sample[(sample > 1)] = 1
                node["output_state"] = sample.reshape(
                    [
                        batch_size,
                    ]
                )

            node["output_state"] = node["output_state"].to(device)
            if node["node_type"] == "biological":
                node["ground_truth"] = node["ground_truth"].to(device)

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
                    node["ground_truth"] = ground_truth[node_name].to(device)

                else:
                    node["ground_truth"] = ground_truth[node_name].to(device)
                    node["output_state"] = ground_truth[node_name].to(
                        device
                    )  # A root node does not need to be predicted

    def propagate_along_edge(self, edge: tuple, inhibition) -> torch.Tensor:
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
        elif self.edges()[edge]["edge_type"] == "simple":
            state_to_propagate = self.nodes[edge[0]]["output_state"]
            return state_to_propagate
        elif self.edges()[edge]["edge_type"] == "transfer_function":
            # The preceding state has to go through the Hill layer
            state_to_propagate = self.edges()[edge]["layer"](
                self.nodes[edge[0]]["output_state"]
            )
        else:
            NameError("The node type is incorrect")
            assert False

        # Include inhibition
        if self.nodes[edge[0]]["node_type"] == "biological":
            state_to_propagate = state_to_propagate / inhibition[edge[0]]
        return state_to_propagate

    def integrate_NOT(self, node: str, inhibition) -> torch.Tensor:
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
            state_to_integrate = self.propagate_along_edge(
                edge=upstream_edges[0], inhibition=inhibition
            )
            ones = torch.ones(state_to_integrate.size())

            ones = ones.to(device)

            result = ones - state_to_integrate
            # TODO check if this is reasonable
            # zeroes = torch.isclose(result, torch.zeros(len(result)))
            # result[zeroes] = 0
            # ones = torch.isclose(result, torch.ones(len(result)))
            # result[ones] = 1
            return result

    def integrate_AND(self, inhibition, node: str) -> torch.Tensor:
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
            self.propagate_along_edge(edge=edge, inhibition=inhibition)
            for edge in upstream_edges
        ]
        # Multiply all the tensors
        result = states_to_integrate[0] * states_to_integrate[1]
        # TODO check if this is reasonable
        # zeroes = torch.isclose(result, torch.zeros(len(result)))
        # result[zeroes] = 0
        # ones = torch.isclose(result, torch.ones(len(result)))
        # result[ones] = 1
        return result

    def integrate_OR(self, inhibition, node: str) -> torch.Tensor:
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
            self.propagate_along_edge(edge=edge, inhibition=inhibition)
            for edge in upstream_edges
        ]

        # Multiply all the tensors
        result = (
            states_to_integrate[0]
            + states_to_integrate[1]
            - states_to_integrate[0] * states_to_integrate[1]
        )
        # TODO check if this is reasonable
        # zeroes = torch.isclose(result, torch.zeros(len(result)))
        # result[zeroes] = 0
        # ones = torch.isclose(result, torch.ones(len(result)))
        # result[ones] = 1
        return result

    def integrate_logical_node(self, node: str, inhibition) -> torch.Tensor:
        """
        A wrapper around integrate_NOT, integrate_OR and integrate_AND to integrate the values
        at any logical node independently of the gate.

        Args:
            - node: the name of the node representing the logical gate
        Returns:
            - The state at the logical gate after integration

        """
        if self.nodes[node]["node_type"] == "logic_gate_AND":
            return self.integrate_AND(node=node, inhibition=inhibition)
        if self.nodes[node]["node_type"] == "logic_gate_OR":
            return self.integrate_OR(node=node, inhibition=inhibition)
        if self.nodes[node]["node_type"] == "logic_gate_NOT":
            return self.integrate_NOT(node=node, inhibition=inhibition)
        else:
            raise NameError(f"{node} is not a known logic gate.")

    def update_biological_node(self, node: str, inhibition) -> torch.Tensor:
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
            return self.propagate_along_edge(
                edge=(parent_node[0], node), inhibition=inhibition
            )
        else:  # For a root edge
            return self.nodes()[node]["ground_truth"]

    def update_fuzzy_node(self, node: str, inhibition) -> None:
        """
        A wrapper to call the correct updating function depending on the type of the node.
        Args:
            - node: name of the node to update
        """

        node_type = self.nodes()[node]["node_type"]
        if node_type == "biological":
            self.nodes()[node]["output_state"] = self.update_biological_node(
                node=node, inhibition=inhibition
            )
        else:
            self.nodes()[node]["output_state"] = self.integrate_logical_node(
                node=node, inhibition=inhibition
            )

    def update_one_timestep_cyclic_network(
        self,
        input_nodes,
        inhibition,
        loop_status,
        convergence_check=False,
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
                    self.update_fuzzy_node(curr_node, inhibition)
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

    def sequential_update(
        self, input_nodes, inhibition, convergence_check=False
    ) -> Optional[dict]:
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

        inhibition = {k: v.to(device) for k, v in inhibition.items()}

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
                        self.update_fuzzy_node(curr_node, inhibition)

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
            length = 20  # 3 * max([len(cycle) for cycle in has_cycle(self)[1]])
            # We simulate length times then output the mean of the last length simulations
            # CHANGED length to int(length/2)
            states[0] = self.output_states
            for i in range(1, int(length)):
                states[i] = self.update_one_timestep_cyclic_network(
                    input_nodes,
                    inhibition,
                    loop_status,
                    convergence_check,
                )
            last_states = {}
            for i in range(int(length)):
                states[length + i] = self.update_one_timestep_cyclic_network(
                    input_nodes,
                    inhibition,
                    loop_status,
                    convergence_check,
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

    def conduct_optimisation(
        self,
        input: dict,
        ground_truth: dict,
        train_inhibitors: dict,
        valid_input: dict,
        valid_ground_truth: dict,
        valid_inhibitors: dict,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optim_wrapper=torch.optim.Adam,
        logger=None,
        convergence_check: bool = False,
        save_checkpoint: bool = True,
        checkpoint_path: str = None,
        patience: int = 20,
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
            - valid_input: dict of torch.Tensor containing root node names mapped to the input validation data
            - valid_ground_truth:  dict of torch.Tensor mapping node names to their value from the validation set
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
            input_nodes = [k for k in valid_input.keys()]
            print(f"There were no root nodes, {input_nodes} were used as input")
        else:
            input_nodes = self.root_nodes

        for node_key, node_tensor in input.items():
            input[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_input.items():
            valid_input[node_key] = node_tensor.to(device)
        for node_key, node_tensor in ground_truth.items():
            ground_truth[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_ground_truth.items():
            valid_ground_truth[node_key] = node_tensor.to(device)
        for node_key, node_tensor in train_inhibitors.items():
            train_inhibitors[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_inhibitors.items():
            valid_inhibitors[node_key] = node_tensor.to(device)

        # Transfer edges (model) to cuda
        for edge in self.transfer_edges:
            self.edges()[edge]["layer"].to(device)

        # Instantiate the dataset
        dataset = DREAMBioFuzzDataset(input, ground_truth, train_inhibitors)
        # Instantiate the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # Keep track of the parameters
        parameters = []
        for edge in self.transfer_edges:
            layer = self.edges()[edge]["layer"]
            parameters += [layer.n, layer.K]

        optim = optim_wrapper(parameters, learning_rate)

        # Train the model
        losses = pd.DataFrame(columns=["time", "loss", "phase"])
        curr_best_val_loss = 1e6
        early_stopping_count = 0

        epoch_pbar = tqdm(range(epochs), desc="Loss=?.??e??")
        train_loss_running_mean = None
        for e in epoch_pbar:
            # Instantiate the model
            self.initialise_random_truth_and_output(batch_size)
            for X_batch, y_batch, inhibited_batch in dataloader:

                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                # Reinitialise the network at the right size
                batch_keys = list(X_batch.keys())
                self.initialise_random_truth_and_output(len(X_batch[batch_keys.pop()]))
                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)
                # Simulate
                loop_states = self.sequential_update(
                    input_nodes,
                    inhibited_batch,
                    convergence_check=convergence_check,
                )

                # Get the predictions
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                # predictions = self.output_states
                labels = {k: v for k, v in y_batch.items() if k in predictions}

                loss = MSE_loss(predictions=predictions, ground_truth=labels)

                # First reset then compute the gradients
                optim.zero_grad()
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_value_(parameters, clip_value=0.5)
                # torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
                # Update the parameters
                optim.step()
                # We save metrics with their time to be able to compare training vs validation
                # even though they are not logged with the same frequency
                if logger is not None:
                    logger.log_metric("train_loss", loss.detach().item())
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
                if train_loss_running_mean is not None:
                    train_loss_running_mean = (
                        0.1 * loss.detach().item() + 0.9 * train_loss_running_mean
                    )
                else:
                    train_loss_running_mean = loss.detach().item()
                epoch_pbar.set_description(f"Loss:{train_loss_running_mean:.2e}")
            # Validation
            with torch.no_grad():
                # Instantiate the model
                self.initialise_random_truth_and_output(
                    len(valid_ground_truth[input_nodes[0]])
                )
                self.set_network_ground_truth(ground_truth=valid_ground_truth)
                # Simulation
                self.sequential_update(input_nodes, valid_inhibitors)
                # Get the predictions
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                labels = {
                    k: v for k, v in valid_ground_truth.items() if k in predictions
                }
                # predictions = self.output_states
                valid_loss = MSE_loss(predictions=predictions, ground_truth=labels)

                # No need to detach since there are no gradients
                if logger is not None:
                    logger.log_metric("valid_loss", valid_loss.item())

                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": valid_loss.item(),
                                "phase": "valid",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

                if curr_best_val_loss > valid_loss:
                    early_stopping_count = 0
                    curr_best_val_loss = valid_loss
                    if checkpoint_path is not None:
                        module_of_edges = torch.nn.ModuleDict(
                            {
                                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                                for edge in self.transfer_edges
                            }
                        )

                        best_model_state = module_of_edges.state_dict()
                        best_optimizer_state = optim.state_dict()

                else:
                    early_stopping_count += 1

                    if early_stopping_count > patience:
                        print("Early stopping.")

                        if checkpoint_path is not None:
                            torch.save(
                                {
                                    "epoch": e,
                                    "model_state_dict": best_model_state,
                                    "optimizer_state_dict": best_optimizer_state,
                                    "loss": valid_loss,
                                    "best_val_loss": curr_best_val_loss,
                                },
                                f"{checkpoint_path}model.pt",
                            )

                            pred_df = pd.DataFrame(
                                {k: v.cpu().numpy() for k, v in predictions.items()}
                            )
                            pred_df.to_csv(
                                f"{checkpoint_path}predictions_with_model_save.csv"
                            )

                        if convergence_check:
                            return losses, curr_best_val_loss, loop_states
                        else:
                            return losses, curr_best_val_loss, None
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": best_optimizer_state,
                        "loss": valid_loss,
                        "best_val_loss": curr_best_val_loss,
                    },
                    f"{checkpoint_path}model.pt",
                )

                pred_df = pd.DataFrame(
                    {k: v.cpu().numpy() for k, v in predictions.items()}
                )
                pred_df.to_csv(f"{checkpoint_path}predictions_with_model_save.csv")

        if convergence_check:
            return losses, curr_best_val_loss, loop_states
        else:
            return losses, curr_best_val_loss, None

    def load_from_checkpoint(self, model_state_dict, model_gate_dict=None):
        module_dict = torch.nn.ModuleDict(
            {
                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                for edge in self.transfer_edges
            }
        )
        module_dict.load_state_dict(model_state_dict)
        edge_att = {
            (k.split("@@@")[0], k.split("@@@")[1]): {"layer": v.to(device)}
            for k, v in module_dict.items()
        }
        nx.set_edge_attributes(self, edge_att)

        if model_gate_dict is not None:
            for node in model_gate_dict.keys():
                self.nodes()[node]["node_type"] = model_gate_dict[node]["node_type"]
                if "gate" in model_gate_dict[node].keys():
                    self.nodes()[node]["gate"] = model_gate_dict[node]["gate"]

    def get_checkpoint(self, save_gates: bool = False):
        module_of_edges = torch.nn.ModuleDict(
            {
                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                for edge in self.transfer_edges
            }
        )

        model_state_dict = module_of_edges.state_dict()

        if save_gates:
            model_gate_dict = {}
            for node in self.nodes:
                model_gate_dict[node] = {}
                model_gate_dict[node]["node_type"] = self.nodes()[node]["node_type"]
                if "gate" in self.nodes()[node].keys():
                    model_gate_dict[node]["gate"] = self.nodes()[node]["gate"]
            return model_state_dict, model_gate_dict
        else:
            return model_state_dict


class DREAMBioFuzzNet(DREAMMixIn, BioFuzzNet):
    def __init__(self, nodes=None, edges=None, n=2, K=0.5):
        super(DREAMBioFuzzNet, self).__init__(nodes, edges, n=n, K=K)

    @classmethod
    def build_DREAMBioFuzzNet_from_file(
        cls, filepath: str, n: float = 2.0, K: float = 0.5
    ):
        """
        An alternate constructor to build the BioFuzzNet from the sif file instead of the lists of nodes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return DREAMBioFuzzNet(nodes, edges, n=n, K=K)


class DREAMBioMixNet(DREAMMixIn, BioFuzzNet):
    def __init__(self, nodes=None, edges=None, AND_param: float = 0.0, n=2, K=0.5):
        super(DREAMBioMixNet, self).__init__(nodes, edges, n=n, K=K)

        for node in self.nodes():
            if self.nodes()[node]["node_type"] in ["logic_gate_AND", "logic_gate_OR"]:
                self.nodes()[node]["node_type"] = "logic_gate_MIXED"
                self.nodes()[node]["gate"] = DREAMMixedGate(
                    AND_param=AND_param,
                    AND_function=self.integrate_AND,
                    OR_function=self.integrate_OR,
                )

    @classmethod
    def build_DREAMBioMixNet_from_file(cls, filepath: str, n=2, K=0.5):
        """
        An alternate constructor to build the BioFuzzNet from the sif file instead of the lists of nodes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return DREAMBioMixNet(nodes, edges, n=n, K=K)

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
            self.nodes()[node_name]["gate"] = DREAMMixedGate(
                AND_param=AND_param,
                AND_function=self.integrate_AND,
                OR_function=self.integrate_OR,
            )

    def integrate_logical_node(self, node: str, inhibition) -> torch.Tensor:
        """
        A wrapper around integrate_NOT, and the MixedGate layer to integrate the different logical nodes.
        Args:
            node: the name of the node representing the logical gate
        Returns:
            The state at the logical gate after integration

        """
        if self.nodes[node]["node_type"] == "logic_gate_AND":
            return self.integrate_AND(node=node, inhibition=inhibition)
        elif self.nodes[node]["node_type"] == "logic_gate_OR":
            return self.integrate_OR(node=node, inhibition=inhibition)
        elif self.nodes[node]["node_type"] == "logic_gate_NOT":
            return self.integrate_NOT(node=node, inhibition=inhibition)
        elif self.nodes[node]["node_type"] == "logic_gate_MIXED":
            return self.nodes[node]["gate"](node=node, inhibition=inhibition)
        else:
            raise NameError("This node is not a known logic gate.")

    def conduct_optimisation(
        self,
        input: dict,
        ground_truth: dict,
        train_inhibitors: dict,
        valid_input: dict,
        valid_ground_truth: dict,
        valid_inhibitors: dict,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optim_wrapper=torch.optim.Adam,
        logger=None,
        save_checkpoint: bool = True,
        checkpoint_path: str = None,
        patience: int = 5,
        loss_weights: float = None,
        mixed_gates_regularisation: float = 1.0,
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
            - valid_input: dict of torch.Tensor containing root node names mapped to the input validation data
            - valid_ground_truth:  dict of torch.Tensor mapping node names to their value from the validation set
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
            input_nodes = [k for k in valid_input.keys()]
            print(f"There were no root nodes, {input_nodes} were used as input")
        else:
            input_nodes = self.root_nodes

        for node_key, node_tensor in input.items():
            input[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_input.items():
            valid_input[node_key] = node_tensor.to(device)
        for node_key, node_tensor in ground_truth.items():
            ground_truth[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_ground_truth.items():
            valid_ground_truth[node_key] = node_tensor.to(device)
        for node_key, node_tensor in train_inhibitors.items():
            train_inhibitors[node_key] = node_tensor.to(device)
        for node_key, node_tensor in valid_inhibitors.items():
            valid_inhibitors[node_key] = node_tensor.to(device)

        # Transfer edges (model) to cuda
        for edge in self.transfer_edges:
            self.edges()[edge]["layer"].to(device)
        for gate in self.mixed_gates:
            gate = self.nodes[gate]["gate"].to(device)

        # Instantiate the dataset
        dataset = DREAMBioFuzzDataset(input, ground_truth, train_inhibitors)

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

        # Set the parameters, leave possibility for other losses/solver
        if loss_weights is None:
            loss_weights = {n: 1 for n in self.biological_nodes}
        optim = optim_wrapper(parameters, learning_rate)

        # Train the model
        losses = pd.DataFrame(columns=["time", "loss", "phase"])
        curr_best_val_loss = 1e6
        early_stopping_count = 0

        epoch_pbar = tqdm(range(epochs), desc="Loss=?.??e??")
        train_loss_running_mean = None
        for e in epoch_pbar:
            # Instantiate the model
            for X_batch, y_batch, inhibited_batch in dataloader:
                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                self.initialise_random_truth_and_output(batch_size)
                # Reinitialise the network at the right size

                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)

                # Simulate
                self.sequential_update(input_nodes, inhibited_batch)

                # Get the predictions
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                labels = {k: v for k, v in y_batch.items() if k in predictions}
                loss = MSE_entropy_loss(
                    predictions=predictions,
                    ground_truth=labels,
                    gates=[self.nodes[node]["gate"] for node in self.mixed_gates],
                    mixed_gates_regularisation=mixed_gates_regularisation,
                )

                # First reset then compute the gradients
                optim.zero_grad()
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_value_(parameters, clip_value=0.5)
                # Update the parameters
                optim.step()
                # We save metrics with their time to be able to compare training vs validation
                # even though they are not logged with the same frequency
                if logger is not None:
                    logger.log_metric("train_loss", loss.detach().item())
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
                if train_loss_running_mean is not None:
                    train_loss_running_mean = (
                        0.1 * loss.detach().item() + 0.9 * train_loss_running_mean
                    )
                else:
                    train_loss_running_mean = loss.detach().item()
                epoch_pbar.set_description(f"Loss:{train_loss_running_mean:.2e}")
            # Validation
            with torch.no_grad():
                # Instantiate the model
                self.initialise_random_truth_and_output(
                    len(
                        valid_ground_truth[input_nodes[0]],
                    )
                )
                self.set_network_ground_truth(ground_truth=valid_ground_truth)
                # Simulation
                self.sequential_update(input_nodes, valid_inhibitors)
                # Get the predictions
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                labels = {
                    k: v for k, v in valid_ground_truth.items() if k in predictions
                }
                # predictions = self.output_states
                valid_loss = MSE_entropy_loss(
                    predictions=predictions,
                    ground_truth=labels,
                    gates=[self.nodes[node]["gate"] for node in self.mixed_gates],
                    mixed_gates_regularisation=mixed_gates_regularisation,
                )

                # No need to detach since there are no gradients
                if logger is not None:
                    logger.log_metric("valid_loss", valid_loss.item())

                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": valid_loss.item(),
                                "phase": "valid",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

                if curr_best_val_loss > valid_loss:
                    early_stopping_count = 0
                    curr_best_val_loss = valid_loss
                    if checkpoint_path is not None:
                        module_of_edges = torch.nn.ModuleDict(
                            {
                                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                                for edge in self.transfer_edges
                            }
                        )

                        best_model_state = module_of_edges.state_dict()
                        best_optimizer_state = optim.state_dict()

                        # torch.save(
                        #     {
                        #         "epoch": e,
                        #         "model_state_dict": best_model_state,
                        #         "optimizer_state_dict": best_optimizer_state,
                        #         "loss": valid_loss,
                        #     },
                        #     f"{checkpoint_path}model.pt",
                        # )

                        # pred_df = pd.DataFrame(
                        #     {k: v.numpy() for k, v in predictions.items()}
                        # )
                        # pred_df.to_csv(f"{checkpoint_path}predictions_with_model.csv")
                else:
                    early_stopping_count += 1

                    if early_stopping_count > patience:
                        print("Early stopping")

                        if checkpoint_path is not None:
                            torch.save(
                                {
                                    "epoch": e,
                                    "model_state_dict": best_model_state,
                                    "optimizer_state_dict": best_optimizer_state,
                                    "loss": valid_loss,
                                    "best_val_loss": curr_best_val_loss,
                                },
                                f"{checkpoint_path}model.pt",
                            )

                            pred_df = pd.DataFrame(
                                {k: v.cpu().numpy() for k, v in predictions.items()}
                            )
                            pred_df.to_csv(
                                f"{checkpoint_path}predictions_with_model_save.csv"
                            )
                        return losses, curr_best_val_loss

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": best_optimizer_state,
                        "loss": valid_loss,
                        "best_val_loss": curr_best_val_loss,
                    },
                    f"{checkpoint_path}model.pt",
                )

                pred_df = pd.DataFrame(
                    {k: v.cpu().numpy() for k, v in predictions.items()}
                )
                pred_df.to_csv(f"{checkpoint_path}predictions_with_model_save.csv")

        return losses, curr_best_val_loss


class DREAMMixedGate(torch.nn.Module):
    """ "
    Implement a MIXED gate which is a linear combination of an AND gate and an OR gate
    MIXED = alpha * AND + (1 - alpha)* OR
    """

    def __init__(self, AND_param: float, AND_function, OR_function) -> None:
        """
        Create the MIXED gate.
        Args:
            - AND_param: sigma^(-1)(alpha) where alpha is the weight of the AND gate in the MIXED gate
            - AND_function: function computed at an AND gate, should be BioMixNet.integrate_AND
            - OR function: function computed at an OR gate, should be BioMixNet.integrate_OR
        """
        torch.nn.Module.__init__(self)
        self.AND_param = torch.nn.Parameter(torch.tensor(AND_param))
        self.AND_function = AND_function
        self.OR_function = OR_function
        self.output_value = None

    def forward(self, node, inhibition):
        """
        Compute the value at the gate.
            Args:
                - node: node at which the input is computed
        """
        AND_value = self.AND_function(node=node, inhibition=inhibition)
        OR_value = self.OR_function(node=node, inhibition=inhibition)
        output = (
            torch.sigmoid(self.AND_param) * AND_value
            + (1 - torch.sigmoid(self.AND_param)) * OR_value
        )
        self.output_value = output
        return output
