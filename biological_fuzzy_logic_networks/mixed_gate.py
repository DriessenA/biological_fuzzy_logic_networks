__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class MixedGate(torch.nn.Module):
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

    def forward(self, x):
        """
        Compute the value at the gate.
            Args:
                - x: node at which the input is computed
        """
        # x is the node at which we integrate the inputs
        AND_value = self.AND_function(node=x)
        OR_value = self.OR_function(node=x)
        output = (
            torch.sigmoid(self.AND_param) * AND_value
            + (1 - torch.sigmoid(self.AND_param)) * OR_value
        )
        self.output_value = output
        return output
