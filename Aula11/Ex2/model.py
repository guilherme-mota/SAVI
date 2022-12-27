#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import torch


# Definition of the model. For now a 1 neuron network
class Model(torch.nn.Module):

    # Constructor
    def __init__(self):

        # Call superclass constructor
        super().__init__()

        # Define the structure of the neural network
        self.layer1 = torch.nn.Linear(1,1)

    # Methods
    def forward(self, xs):

        ys = self.layer1(xs)

        return ys