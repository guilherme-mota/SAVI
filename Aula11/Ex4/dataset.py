#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, num_points, f=1, a=2, sigma=1.4):

        # Call superclass constructor
        super().__init__()

        self.num_points = num_points

        # Genarate Data
        self.xs_np = np.random.rand(num_points, 1)*20 - 10  # to get values between -10 e 10
        self.xs_np = self.xs_np.astype(np.float32)
        self.ys_np_labels = np.sin(f * self.xs_np) * a  # Compute ys
        self.ys_np_labels += np.random.normal(loc=0.0, scale=sigma, size=(num_points,1))  # add noise

        # Convert to torch tensor
        self.xs_ten = torch.from_numpy(self.xs_np)
        self.ys_ten = torch.from_numpy(self.ys_np_labels)

    # Methods
    def __getitem__(self, index):

        # Returns a specific element (x,y), given the index of the dataset
        return self.xs_np[index], self.ys_np_labels[index]

    def __len__(self):
        
        # Return the length of the dataset
        return self.num_points