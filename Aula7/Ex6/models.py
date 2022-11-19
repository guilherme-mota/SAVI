#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import math
import numpy as np
from random import uniform
import matplotlib.pyplot as plt


class Plynomial():
    """Defines the model of a polinomial function"""

    def __init__(self, gt):
        self.gt = gt
        self.first_draw = True
        self.randomizeParams()
        self.xs_for_plot = list(np.linspace(-10, 10, num=500))

    def randomizeParams(self):
        self.a = uniform(-1, 1)
        self.b = uniform(-1, 1)
        self.c = uniform(-1, 1)
        self.d = uniform(-1, 1)
        self.e = uniform(-1, 1)
        self.f = uniform(-1, 1)
        self.g = uniform(-1, 1)
        self.h = uniform(-1, 1)

    def getY(self, x):
        return  self.a + self.b*x + self.c*math.pow(x,2) + self.d*math.pow(x,3)  + self.e*math.pow(x,4) + self.f*math.pow(x,5) + self.g*math.pow(x,6) + self.h*math.pow(x,7) 

    def getYs(self, xs):
        """Retrieves a list of ys by applying the model to a list of xs"""
        ys = []
        for x in xs:
            ys.append(self.getY(x))

        return ys

    def objectiveFunction(self, params):
        # Convert scipy params list into class params
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.e = params[4]
        self.f = params[5]
        self.g = params[6]
        self.h = params[7]

        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x)  # compute Y using the current model parameters

            # difference between real Y and computed value
            residual = abs(y - gt_y)
            residuals.append(residual)

        # draw for visualization
        self.draw()
        plt.waitforbuttonpress(0.1)

        return residuals

    def draw(self, color='b--'):
        xi = -10
        xf = 10
        yi = self.getY(xi)
        yf = self.getY(xf)

        # Verify if the plot was already drawn
        if self.first_draw:
            self.draw_handler = plt.plot(self.xs_for_plot, self.getYs(self.xs_for_plot), color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handler, data=(self.xs_for_plot, self.getYs(self.xs_for_plot)))  # update plot