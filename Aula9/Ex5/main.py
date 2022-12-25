#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import open3d as o3d
import numpy as np
from copy import deepcopy
from random import randint
from matplotlib import cm
from more_itertools import locate


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


def main():

    #-----------------
    # Initialization
    #-----------------
    print("Load a ply point cloud, print it, and render it")
    point_cloud_original = o3d.io.read_point_cloud('../Ex4/factory_isolated.ply')
    #
    

    #-----------------
    # Execution
    #-----------------
    point_cloud = deepcopy(point_cloud_original)
    
    # Estimate Normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    #-------------------
    # Visualization
    #-------------------
    entities = [point_cloud]  # createa list of entities to draw

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities,
                                        zoom = view['trajectory'][0]['zoom'],
                                        front = view['trajectory'][0]['front'],
                                        lookat = view['trajectory'][0]['lookat'],
                                        up = view['trajectory'][0]['up'],
                                        point_show_normal=True)

  
if __name__ == '__main__':
    main()