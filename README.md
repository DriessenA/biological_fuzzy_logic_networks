# Biological Fuzzy logic Networks (BioFuzzNet)

Fuzzy Logical Networks for Biological Systems. Optimized with stochastic gradient descent.

## Environment
Using a virtual environment for all commands in this guide is strongly recommended.

## Installation

### Package installation
1. Create virtual environment with python 3.10
2. Clone package
3. Go into package directory

```sh
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Usage
...

## Contributing

Check [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Getting support

Check [SUPPORT.md](.github/SUPPORT.md).

# Technical details

The intent of this section is to provide a description of how a BioFuzzNet is implemented and why those implementation choices were made.

A more precise description of the algorithms is available in the MSc thesis of Constance Le Gac at ETHZ (link when available).

## General description of a BioFuzzNet

A BioFuzzNet is an object encoding a Boolean network (assumed to represent a biological system, although any Boolean network can be used) and supporting a fuzzy logic formalism.

### Graphical structure of a BioFuzzNet

A BioFuzzNet is at its core a networkx DiGraph with different types of nodes and edges.
Nodes can be of four types:
    * Biological nodes represent biological entities, or "real" nodes, that is the species that are modelled
    * Logical nodes can be of three types: AND, OR, or NOT. The logical operations correspond to the fuzzy equivalent of the usual Boolean logic operations. More information on those operations can be found by looking for "fuzzy logic operators" or "Zadeh logic operators".
        * NOT gates must have an in-degree of one and implement a fuzzy logic negation given by output = 1 - input.
        * AND gates must have an in-degree of two and implement a fuzzy logic conjunction. If i1 and i2 are the two inputs, the output of the AND gate is given by $output = i1 * i2$
        * OR gates must have an in-degree of two and implement a fuzzy logic disjunction. If i1 and i2 are the two inputs, the output of the OR gate is given by $output = (i1 + i2)- i1 * i2$

Edges can be of two types. There are always positive, as negations are encoded through the NOT gates:
    * Simple edges which merely transmit the output state of the updstream node to the downstream node. Those always have a logical node as an upstream node and can have a logical or biological node as a downstream node
    * Transfer edges which transform the output of the upstream node through a transfer function before transmitting it to the downstream node. Those always have a biological node as an upstream node, and can have a logical or biological node as an upstream node.
The rationale behind this choice is that the Hill functions model the way the biological entity will act on other biological entities.

### Structure of the Hill functions

Non-normalized Hill functions were chosen to act as transfer functions. Hill functions were chosen at they are known to be acceptable mathematical representations for a range of biological phenomena.
Each transfer function has two parameters:
    * n is the Hill cooperativity coefficient. It represents the steepness of the curve and is always positive.
    * K is the EC50 coefficient and represent the concentration of input at which 50 \% of the maximum response is used.
Due to the shape of and parameterization of the Hill function, $\forall n, K: Hill(0, n, K) = 0 \text{ and } Hill(1, n, K) = \frac{1}{1 + K^n}$. Hence by manipulating the value of K and n it is possiblel to get a value of Hill(1, n, K) arbitrarily close to 0 and thus "turn the edge OFF".
It would also be possible to add and tune a scale parameter. However this seemed to reduce the identifiability of the model and was rejected.

### Value at the nodes
Each biological nodes is associated to 2 tensors: the output_state tensor contains the current predicted values for all simulated cells at this node, whereas the ground_truth tensor contains the measured values for all simulated cells, provided the node is measured.
As logical nodes cannot be measured, they only contain an output_state tensor.

## Simulation and Optimisation of a BioFuzzNet

The parameters of the Hill functions are tuned using a gradient-descent based optimisation. As an automatic differentiation engine is used to compute ythe gradient, it is necessary that all operations during the simulation and optimisation process stay differentiable. This is why the BioFuzzNet is simulated using a sequential update mechanism in which the network is traversed from root nodes to leaf nodes.
The simulation process needs as input the values of the most upstream nodes in the network, and at least all the root nodes. Other nodes are then updated after their parents, in a manner similar to a Breadth-First Search. In cases where the network contains a cycle, at least one node in the cycle will need to be updated before all of its parents are updated. In this case, the loop is iterated through several times. An average of the last k predictions (where k is proportional to the length of the largest loop in the network) is then used as the predicted value.

ADAM is the chosen optimiser, and the loss function is the mean squared error between predicted and observed values at nodes.
The loss function is given by $Loss(B,b, I, y) = \frac{1}{b} \sum_{k = 1}^{b} \frac{1}{n}\sum_{j = 1}^n (B(I)_j - y_j)^2$ where $B$ refers to the BioFuzzNet, $b$ is the batch size,  $I$ is the input given to the model, $y$ is the vector of measured values, $B(I)_j$ is the predicted value at measured node $j$ given input $I$ and $y_j$ is the measured value at this node.


## General definition of a BioMixNet

The BioMixNet class is a class extending on the BioFuzzNet module, which allows for undecided logical gates called MIXED gates. Those gates are a combination of an OR and an AND gate: the operation on them is given by:
$f_{MIXED} = \alpha \cdot f_{AND} + (1-\alpha) \cdot \f_{OR}$

The $\alpha$ parameter is also included in the list of parameters to optimise during the gradient descent. The loss is however modified by a regularisation term that ensures the $\alpha$ parameter takes a value close to 0 or 1 after optimisation, therefore clearly identifying the gate as an OR or an AND gate. The modified value of the loss function is then:
$Loss(B,b, I, y, \{\alpha\}_{i=1}^m) = \frac{1}{b} \sum_{k = 1}^{b} \frac{1}{n}\sum_{j = 1}^n (B(I)_j - y_j)^2  + C_{reg} \cdot \sum_{i=1}^m \alpha_i \cdot (1 - \alpha_i)$
where $\{\alpha\}_{i=1}^m$ are the MIXED gate coefficients, and $C_{reg}$ is a regularisation constant, set to 1 by default.


After optimisation, the value of the $\alpha$ parameter indicates the type of the optimised gate: if it's 1 then it's an AND gate, it it's 0 then it's an OR gate, if it's close to neither of those then the gate could not be tuned.
