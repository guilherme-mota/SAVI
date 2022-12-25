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
    point_cloud = o3d.io.read_point_cloud('/home/guilherme/savi_22-23/Parte09/data/Factory/factory.ply')
    

    #-----------------
    # Execution
    #-----------------
    print('Starting plane detection')
    plane_model, inlier_idxs = point_cloud.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=150)
    [a, b, c, d] = plane_model
    print('Plane equation: ' + str(a) + ' x + ' + str(b) + ' y + ' + str(c) + ' z + ' + str(d) + ' = 0')

    inlier_cloud = point_cloud.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color([1,0,0])  # paints the plane in red
    outlier_cloud = point_cloud.select_by_index(inlier_idxs, invert=True)


    #-------------------
    # Visualization
    #-------------------
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                        zoom = view['trajectory'][0]['zoom'],
                                        front = view['trajectory'][0]['front'],
                                        lookat = view['trajectory'][0]['lookat'],
                                        up = view['trajectory'][0]['up'])


    #-----------------
    # Termination
    #-----------------
    o3d.io.write_point_cloud('./factory_without_ground.ply', outlier_cloud, write_ascii=False, compressed=False, print_progress=False)
    

if __name__ == '__main__':
    main()