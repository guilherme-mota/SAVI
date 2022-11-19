#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import pickle
import numpy as np
from models import Line
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
    line = Line(pts)


    #-----------------
    # Execution
    #-----------------

    # Set new values
    line.randomizeParams()

    least_squares(line.objectiveFunction, [line.m, line.b], verbose=2)


    #-----------------
    # Termination
    #-----------------


if __name__ == '__main__':
    main()