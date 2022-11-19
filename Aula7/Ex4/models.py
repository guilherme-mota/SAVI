#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
from random import uniform
import matplotlib.pyplot as plt


class Line():
    """Defines the model of a line segment"""

    def __init__(self, gt):
        self.gt = gt
        self.first_draw = True
        self.randomizeParams()

    def randomizeParams(self):
        self.m = uniform(-2, 2)
        self.b = uniform(-5, 5)

    def getY(self, x):
        return self.m * x + self.b

    def objectiveFunction(self, params):
        self.m = params[0]
        self.b = params[1]

        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x)  # compute Y using the current model parameters

            # difference between real Y and computed value
            residual = abs(y - gt_y)
            residuals.append(residual)

        # the error is the sum of the residuals
        error = sum(residuals)
        print('error = ' + str(error))

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
            self.draw_handler = plt.plot([xi, xf], [yi, yf], color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handler, data=([xi, xf], [yi, yf]))  # update plot