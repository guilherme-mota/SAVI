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
    point_cloud_original = o3d.io.read_point_cloud('../Ex2/factory_without_ground.ply')
    

    #-----------------
    # Execution
    #-----------------
    point_cloud = deepcopy(point_cloud_original)
    print('before downsampling point cloud has ' + str(len(point_cloud.points)) + ' points')

    # Downsampling using voxel grid filter
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.1) 
    print('After downsampling point cloud has ' + str(len(point_cloud_downsampled.points)) + ' points')

    # Clustering ------------------------------
    cluster_idxs = list(point_cloud_downsampled.cluster_dbscan(eps=0.45, min_points=50, print_progress=True))

    print(cluster_idxs)
    print(type(cluster_idxs))

    possible_values = list(set(cluster_idxs))
    possible_values.remove(-1)
    print(possible_values)

    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))

    cloud_building = point_cloud_downsampled.select_by_index(largest_idxs)
    cloud_others = point_cloud_downsampled.select_by_index(largest_idxs, invert=True)

    cloud_others.paint_uniform_color([0,0,0.5])

    #-------------------
    # Visualization
    #-------------------
    entities = [cloud_building, cloud_others]  # createa list of entities to draw

    o3d.visualization.draw_geometries(entities,
                                        zoom = view['trajectory'][0]['zoom'],
                                        front = view['trajectory'][0]['front'],
                                        lookat = view['trajectory'][0]['lookat'],
                                        up = view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('./factory_isolated.ply', cloud_building, write_ascii=False, compressed=False, print_progress=False)

  

if __name__ == '__main__':
    main()