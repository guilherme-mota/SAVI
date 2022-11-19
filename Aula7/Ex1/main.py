#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():

    #-----------------
    # Initialization
    #-----------------
    plt.figure()
    plt.grid()
    plt.xlim(-10,10)
    plt.xlabel("X")
    plt.ylim(-10,10)
    plt.ylabel("Y")
    
    print('Created a figure!')

    # dictionary with a list of X values and Y values
    pts = {'xs': [], 'ys': []}


    #-----------------
    # Execution
    #-----------------
    while True:

        # Plot values in the pts dictionary
        plt.plot(pts['xs'], pts['ys'], 'rx', linewidth=2, markersize=12)

        # Get user input
        pt = plt.ginput(1)

        # Verify if the list is empty
        if not pt:
            print('Terminated!')
            break

        print('pt = '+ str(pt))

        # Add values to the pts dictionary
        pts['xs'].append(pt[0][0])
        pts['ys'].append(pt[0][1])

        print('pts = ' + str(pts))


    #-----------------
    # Termination
    #-----------------
    file = open('pts.pkl', 'wb')
    pickle.dump(pts, file)
    file.close()


if __name__ == '__main__':
    main()