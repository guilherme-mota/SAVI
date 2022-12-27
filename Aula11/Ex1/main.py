#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


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


def main():

    #-----------------------------------------------------------------
    # Initialization
    #-----------------------------------------------------------------

    # Read file with points
    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()
    print('pts = ' + str(pts) + '\n')  # print file information

    # Convert the pts into np arrays
    xs_np = np.array(pts['xs'], dtype=np.float32).reshape(-1,1)
    ys_np_labels = np.array(pts['ys'], dtype=np.float32).reshape(-1,1)

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

        xs_ten = torch.from_numpy(xs_np).to(device)
        ys_ten_labels = torch.from_numpy(ys_np_labels).to(device)

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
    ys_ten_predicted = model.forward(xs_ten)
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    # Plot the values
    plt.plot(xs_np, ys_np_labels,'g.', label = 'labels')
    plt.plot(xs_np, ys_np_predicted,'rx', label = 'predicted')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()