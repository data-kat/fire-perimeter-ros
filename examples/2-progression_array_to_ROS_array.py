#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:26:36 2021

@author: Kate, Benjamin
"""



import numpy as np
import os

from functions import base
from functions import ros

wd_path = os.getcwd()
datapath = os.path.join(wd_path, "data")


# Read in the ground control points (gcps) file, convert to a GDAL gcps object,
# and use the gcps object to obtain the geotransformation string
gcps_p1_filename = os.path.join(datapath, "links_plot1.txt")
gcps_p1 = base.gcps_from_links_file(gcps_p1_filename)
geotransform_p1 = base.geotransform_from_gcps(gcps_p1)


# Load in the fire progression array obtained from processing the whole video
#  in teh earlier step:
prog_arr_p1 = np.load(os.path.join(datapath, "prog_arr_3fps_p1-2021-03-01.npy"),
                      allow_pickle=True)

# Can view the array using the commented-out line below
# base.imshow_colorbar(prog_arr_p1)


# Calculate ROS

ros_array_xy_p1 = ros.get_ros_from_progress_array(
        progress_array = prog_arr_p1,
        levels = 20,
        output = "x_y",
        max_interval = None)

# Can view the array using the commented-out line below
# base.imshow_colorbar(ros_array_xy_p1[:, :, 0]) # x-axis ROS
# base.imshow_colorbar(ros_array_xy_p1[:, :, 1]) # y-axis ROS


np.save(os.path.join(datapath, "ros_array_xy_p1.npy"), ros_array_xy_p1)


# if the above takes too long and you want to skip it, load in the precalculated raster
ros_array_xy_p1 = np.load(os.path.join(datapath, "ros_array_xy_p1.npy"),
                          allow_pickle=True)



# Convert image-based x and y vectors to a geographical vectors based on
#   the geotransformation

ros_array_xy_p1_ms = ros.xy_array_ppf_to_ms(ros_array_xy_p1, geotransform_p1, fps = 3)

base.imshow_colorbar(ros_array_xy_p1_ms[:, :, 0])

ros_array_rosangle_p1_ms = ros.xy_to_rosangle(ros_array_xy_p1_ms)

base.imshow_colorbar(ros_array_rosangle_p1_ms[:, :, 0])



### mask out pixels outside the fire polygon:

p1_poly_shp = os.path.join(datapath, "rgb_active_extent_p1.shp")

p1_mask = base.polygon_mask_from_shapefile(p1_poly_shp, ros_array_rosangle_p1_ms, geotransform_p1)

ros_array_rosangle_p1_ms_masked = ros_array_rosangle_p1_ms.copy()

ros_array_rosangle_p1_ms_masked[~p1_mask] = np.nan

base.imshow_colorbar(ros_array_rosangle_p1_ms_masked[:, :, 0])


