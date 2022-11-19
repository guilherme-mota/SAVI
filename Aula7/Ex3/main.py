#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import pickle
import numpy as np
from models import Sinusoid
import matplotlib.pyplot as plt
from colorama import Fore, Style


def main():

    #-----------------
    # Initialization
    #-----------------

    # Load file with points
    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()
    print('pts = ' + str(pts))

    # Create figure
    plt.figure()
    plt.grid()
    plt.xlim(-10,10)
    plt.xlabel("X")
    plt.ylim(-10,10)
    plt.ylabel("Y")
    print('Created a figure!')

    # Draw ground truth pts
    plt.plot(pts['xs'], pts['ys'], 'sk', linewidth=2, markersize=6)

    # Define the model
    model = Sinusoid(pts)
    best_model = Sinusoid(pts)
    best_error = 1E6  # a very larger number to start


    #-----------------
    # Execution
    #-----------------
    while True:  # Iterate setting new values for the params and recomputing the error

        # Set new values
        model.randomizeParams()

        # Compute error
        error = model.objectiveFunction()
        print(error)

        # Verify if the model is better
        if error < best_error:

            print(Fore.RED + 'We found a better model!' + Style.RESET_ALL)

            # Update variables
            best_model.a = model.a
            best_model.b = model.b
            best_model.h = model.h
            best_model.k = model.k
            best_error = error

        # Draw line models
        model.draw()
        best_model.draw('r-')

        plt.waitforbuttonpress(0.1)

        if not plt.fignum_exists(1):  # a way to do clean termination
            print('Terminating!')
            break


    #-----------------
    # Termination
    #-----------------


if __name__ == '__main__':
    main()