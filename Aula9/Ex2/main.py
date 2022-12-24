#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import open3d as o3d
import numpy as np


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
    ply_point_cloud = o3d.data.PLYPointCloud()
    

    #-----------------
    # Execution
    #-----------------
    point_cloud = o3d.io.read_point_cloud('/home/guilherme/savi_22-23/Parte09/data/Factory/factory.ply')
    print(point_cloud)
    print(np.asarray(point_cloud.points))
    o3d.visualization.draw_geometries([point_cloud],
                                        zoom = view['trajectory'][0]['zoom'],
                                        front = view['trajectory'][0]['front'],
                                        lookat = view['trajectory'][0]['lookat'],
                                        up = view['trajectory'][0]['up'])


    #-----------------
    # Termination
    #-----------------
    

if __name__ == '__main__':
    main()