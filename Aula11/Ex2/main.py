#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import torch
import pickle
import numpy as np
from model import Model
from dataset import Dataset
import matplotlib.pyplot as plt


def main():

    #-----------------------------------------------------------------
    # Initialization
    #-----------------------------------------------------------------

    # Create the Dataset
    dataset = Dataset(3000, 0.3, 14)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    # Draw training data
    plt.plot(dataset.xs_np, dataset.ys_np_labels,'g.', label = 'labels')
    plt.legend(loc='best')
    plt.show()

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # cuda: 0 index of gpu

    model = Model()  # instantiate Model
    model.to(device)  # move th model variable to the gpu if one exists

    # Variables of the process
    learning_rate = 0.01
    maximum_num_epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    idx_epoch = 0
    while True:

        # Train batch by batch
        for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):

            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)

            # Apply the network to get the predicted ys
            ys_ten_predicted = model.forward(xs_ten)

            # Compute de error based on the predictions
            loss = criterion(ys_ten_predicted, ys_ten_labels)

            # Update the model, i.e. the neural network's weights
            optimizer.zero_grad()  # resets the weights to make sure we aren't accumulating
            loss.backward() # propagates the loss error each neuron
            optimizer.step() # update the weights

            # Report
            print('Epoch ' + str(idx_epoch) + ', Loss ' + str(loss.item()))

        # Increment Epoch count
        idx_epoch += 1

        # Termination Criteria
        if idx_epoch > maximum_num_epochs:
            print('\nFinished training. Reached maximum number of epochs!\n')

            break


    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    # Run the Model once to get ys_predicted
    ys_ten_predicted = model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    # Plot the result and training values
    plt.plot(dataset.xs_np, dataset.ys_np_labels,'g.', label = 'labels')
    plt.plot(dataset.xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()