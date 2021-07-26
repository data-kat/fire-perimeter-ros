# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:33:32 2021

@author: kme96
"""

import os

wd_path = r"C:\Users\kme96\OneDrive - University of Canterbury\Fire-front-edge-detection-new\VFPT"
os.chdir(wd_path)


from functions import base


datapath = os.path.join(wd_path, "data/")



video_fn_p1 = datapath+"/P1_3fps_500_to_521.mp4"


video_p1 = base.VideoExtractor(video_fn_p1)


base.find_ideal_range_video(video_fn_p1, 3)



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



# press Esc to exit
base.check_ranges_on_video(video_fn_p1, range_list_p1)




# Reduce the frame rate of a video


video_p1.save_video_low_fps(new_fps = None, stop_at_frame = None)

















