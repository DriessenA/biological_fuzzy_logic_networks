"""A Pytorch module defining a Hill function"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import torch
from math import log

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HillTransferFunction(torch.nn.Module):
    """Apply a Hill transformation on 1D input"""

    def __init__(self, K=0.5, n=2):
        """
        Initialise the parameters if the transfer functino
        """
        torch.nn.Module.__init__(self)

        self.n = torch.nn.Parameter(torch.normal(mean=log(n - 1), std=0.4, size=(1,)))
        self.K = torch.nn.Parameter(torch.normal(mean=log(K), std=0.2, size=(1,)))

    def forward(self, x):
        """
        Tranforms a value through the transfer function
        Args:
            x = value to be transformed
        """
        # Hill function as in Eduati et al. Cancer Research 2017
        # https://doi.org/10.1158/0008-5472.CAN-17-0078
        K = torch.exp(self.K)  # Ensure non-negative parameters
        n = 1 + torch.exp(self.n)  # 1 + n to ensure n > 1
        x_min = 1 - x

        output = (x_min**n) / (K**n + x_min**n)
        # in the normalized_Hill branch we normalize the outputs
        output = output * (1 + K**n)
        self.output_value = 1 - output
        return self.output_value
