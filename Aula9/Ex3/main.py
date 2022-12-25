#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import open3d as o3d
import numpy as np
from copy import deepcopy
from random import randint
from matplotlib import cm


view = {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
                    "boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.87841585716373893, 0.25915540896962569, 0.40152715460486627 ],
                    "lookat" : [ -9.2636016725183961, 4.4890530163372029, -4.8087575728461065 ],
                    "up" : [ 0.36915599662500914, -0.16561754707620108, 0.91449148615843245 ],
                    "zoom" : 0.60120000000000018
                }
            ],
            "version_major" : 1,
            "version_minor" : 0
        }


class PlaneDtection():

    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):

        self.inlier_cloud.paint_uniform_color([r, g, b])  # paints the palne

    def segment(self, distance_threshold=0.25, ransac_n=3, num_iterations=100):

        print('Starting plane detection')

        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)

        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) + ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0'

        return text


def main():

    #-----------------
    # Initialization
    #-----------------
    print("Load a ply point cloud, print it, and render it")
    point_cloud_original = o3d.io.read_point_cloud('../Ex2/factory_without_ground.ply')

    number_of_planes = 6
    colormap = cm.Pastel1(list(range(0, number_of_planes)))  # color map for planes
    

    #-----------------
    # Execution
    #-----------------
    point_cloud = deepcopy(point_cloud_original)
    planes = []

    while True:  # run consecutive plane detections

        plane = PlaneDtection(point_cloud)  # create a new plane instance
        point_cloud = plane.segment()  # new point cloud are the outliers of this plane detection
        print(plane)

        # colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        plane.colorizeInliers(color[0], color[1], color[2])

        planes.append(plane)

        if len(planes) >= number_of_planes:  # stop detection planes
            break


    #-------------------
    # Visualization
    #-------------------
    entities = [x.inlier_cloud for x in planes]  # createa list of entities to draw
    entities.append(point_cloud)

    o3d.visualization.draw_geometries(entities,
                                        zoom = view['trajectory'][0]['zoom'],
                                        front = view['trajectory'][0]['front'],
                                        lookat = view['trajectory'][0]['lookat'],
                                        up = view['trajectory'][0]['up'])
  

if __name__ == '__main__':
    main()