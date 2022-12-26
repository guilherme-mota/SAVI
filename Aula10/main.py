#!/usr/bin/env python3


#-----------------
# Imports
#-----------------
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from point_cloud_processing import PointCloudProcessing


view = {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 0.90000000000000002, 0.90000000000000002, 0.5 ],
                    "boundingbox_min" : [ -0.90000000000000002, -0.90000000000000002, -0.29999999999999999 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.53546603193045594, 0.64498881755287318, 0.54522064696817796 ],
                    "lookat" : [ 0.33718259194605543, 0.53599004839044462, -0.011415990610460006 ],
                    "up" : [ -0.33196180259186531, -0.43287342813100638, 0.83810617277172583 ],
                    "zoom" : 0.5199999999999998
                }
            ],
            "version_major" : 1,
            "version_minor" : 0
        }


def main():

    #-----------------
    # Initialization
    #-----------------
    p = PointCloudProcessing()
    p.loadPointCloud('/home/guilherme/savi_22-23/Parte10/data/scene.ply')


    #-----------------
    # Execution
    #-----------------
    # Exercício 1
    p.preProcess(voxel_size=0.01)


    # Exercício 2
    p.transform(-108,0,0,0,0,0)  # rotate around Z
    p.transform(0,0,-37,0,0,0)  # rotate around X
    p.transform(0,0,0,-0.85,-1.10,0.35)  # translate


    # Exercício 3
    p.crop(-0.9, -0.9, -0.3, 0.9, 0.9, 0.4)  # bounding box values


    # Exercício 4
    outliers = p.findPlane()  # returns points outside the plane of the table


    # Exercício 5 - Clustering
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=60, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0, number_of_objects)))

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)

        # Create a Dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color'])  # paints the plane in red
        d['center'] = d['points'].get_center()

        objects.append(d)  # Add the  dictionary of this object to the objects list


    # Exercício 6 - ICP
    cereal_box_model = o3d.io.read_point_cloud('/home/guilherme/savi_22-23/Parte10/data/cereal_box_2_2_40.pcd')

    for object_idx, object in enumerate(objects):

        print('Apply poit-to-point ICP to object ' + str(object['idx']))

        trans_init = np.asarray([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])

        reg_p2p = o3d.pipelines.registration.registration_icp(cereal_box_model, 
                                                              object['points'], 2, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())

        print(reg_p2p.inlier_rmse)
        object['rmse'] = reg_p2p.inlier_rmse

    # How to classify the object. Use the smallest fitting to decide which object is a "cereal box"
    minimum_rmse = 10e8  # just a very large number to start
    cereal_box_object_idx = None

    for object_idx, object in enumerate(objects):

        if object['rmse'] < minimum_rmse:  # Found a new minimum
            minimum_rmse = object['rmse']
            cereal_box_object_idx = object_idx

    print('The cereal box is object ' + str(cereal_box_object_idx))


    #-------------------
    # Visualization
    #-------------------
    entities = [p.pcd]  # createa list of entities to draw

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw BoundingBox --------------------------------------
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    # Draw Objects --------------------------------------
    for object in objects:
        entities.append(object['points'])

    # --------------------------------------------------------------------
    # Make a more complex open3D window to show object labels on top of 3D
    # --------------------------------------------------------------------
    app = gui.Application.instance
    app.initialize() # create an open3d app

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  # set black background

    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2 * w.scaling

    # Draw entities
    for entity_idx, entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)

    # Draw labels
    for object_idx, object in enumerate(objects):
        label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.15]

        label_text = object['idx']
        if object_idx == cereal_box_object_idx:
            label_text += ' (Cereal Box)'

        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(object['color'][0], object['color'][1],object['color'][2])
        label.scale = 2
        
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()


if __name__ == '__main__':
    main()