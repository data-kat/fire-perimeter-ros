#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:21:48 2021

@author: Kate, Benjamin
"""





import os
import numpy as np

wd_path = r"C:\Users\kme96\OneDrive - University of Canterbury\Fire-front-edge-detection-new\VFPT"
os.chdir(wd_path)

from functions import base


datapath = os.path.join(wd_path, "data/")



##############################################################################
#############          SPECIFY NECESSARY FILENAMES            ################
##############################################################################


video_fn_p1 = datapath+"/P1_3fps_500_to_521.mp4"


video_p1 = base.VideoExtractor(video_fn_p1)




##############################################################################
##############       COLOR THRESHOLDING          ###############
##############################################################################


black1_p1 = [(16, 0, 0), (163, 255, 67)]
black2_p1 = [(29, 67, 108), (255, 255, 255)]
black3_p1 = [(56, 36, 59), (143, 153, 169)]
black4_p1 = [(72, 6, 17), (163, 255, 67)]

flame1_p1 = [(0, 130, 147), (13, 255, 255)]
flame2_p1 = [(0, 92, 133), (9, 207, 169)]
flame3_p1 = [(0, 57, 147), (8, 144, 190)]
flame4_p1 = [(4, 61, 137), (6, 117, 209)]

range_list_p1 = [black1_p1, black2_p1, black3_p1, black4_p1,
                 flame1_p1, flame2_p1, flame3_p1, flame4_p1]



##############################################################################
##############       PROGRESSION RASTER EXTRACTION          ###############
##############################################################################


prog_arr_p1, prog_cnts_list_p1 = base.get_progress_array_and_cnts_list(
        video_fn_p1,
        range_list_p1,
        blur = None)

np.save(datapath+"P1_3fps_progRaster_500_to_521.npy", prog_arr_p1)


# If wanted the progression array can be plotted with:
# base.imshow_colorbar(prog_arr_p1)



















