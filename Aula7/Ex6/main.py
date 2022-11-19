#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import pickle
import numpy as np
from models import Plynomial
import matplotlib.pyplot as plt
from colorama import Fore, Style
from scipy.optimize import least_squares


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
    model = Plynomial(pts) 


    #-----------------
    # Execution
    #-----------------

    # Set new values
    model.randomizeParams()

    x0 = [model.a, model.b, model.c, model.d, model.e, model.f, model.g, model.h]
    least_squares(model.objectiveFunction, x0, verbose=2)


    #-----------------
    # Termination
    #-----------------
    model.draw()
    plt.show()


if __name__ == '__main__':
    main()