# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:02:00 2020

@author: K Melnik


Module docstrings are similar to class docstrings. Instead of classes and class methods being documented, it’s now the module and any functions found within. Module docstrings are placed at the top of the file even before any imports. Module docstrings should include the following:

    A brief description of the module and its purpose
    A list of any classes, exception, functions, and any other objects exported by the module

The docstring for a module function should include the same items as a class method:

    A brief description of what the function is and what it’s used for
    Any arguments (both required and optional) that are passed including keyword arguments
    Label any arguments that are considered optional
    Any side effects that occur when executing the function
    Any exceptions that are raised
    Any restrictions on when the function can be called


"""


import math
import numpy as np


def get_angle_from_displacement(x, y):
    degrees_raw = np.rad2deg(np.arctan2(x, y))
    degrees_positive = degrees_raw % 360
    return(degrees_positive)


def limit_timediff_array_to_adjacent(timediff_array, max_interval):
    
    if np.isnan(timediff_array).all():
        array_of_nans = np.empty(timediff_array.shape)
        array_of_nans[:] = np.nan
        return array_of_nans    
    
    # Find the smallest absolute value and check that it is not greater than max_interval
    min_abs_val = np.min(abs(timediff_array[~np.isnan(timediff_array)]))
    
    if max_interval is None or min_abs_val <= max_interval:
        timediff_array_out = timediff_array.copy()
        timediff_array_out[abs(timediff_array_out) != min_abs_val] = np.nan
        # print(f"deleted non-{interval} values")
        return timediff_array_out
    
    else:
        array_of_nans = np.empty(timediff_array.shape)
        array_of_nans[:] = np.nan
        return array_of_nans




def get_mean_ros(array, output, max_interval):
    
    if output not in ("ros_angle", "x_y"):
        raise Exception(f'output set to "{output}", please set to "ros_angle" or "x_y"')
    
    
    if np.isnan(array).all():
        return (np.nan, np.nan)
    
    
    center_row = math.floor(array.shape[0]/2)
    center_col = math.floor(array.shape[1]/2)
    
    if np.isnan(array[center_row, center_col]):
        return (np.nan, np.nan)
    
    timediff_array = (array - array[center_row, center_col]).astype(float)
    timediff_array[timediff_array == 0] = np.nan
    
    
    limited_timediff_array = limit_timediff_array_to_adjacent(
        timediff_array = timediff_array,
        max_interval = max_interval)
    
    
    rows, cols = np.where(~np.isnan(limited_timediff_array))
    
    ros_x_list = [None] * len(rows)
    ros_y_list = [None] * len(rows)
    
    for i, (r, c) in enumerate(zip(rows, cols)):
        # print(i, r, c)
        
        dist_x = c - center_col
        dist_y = center_row - r
        # dist = math.sqrt(dist_y**2 + dist_x**2)
        
        timediff = limited_timediff_array[r, c]
        
        ros_x = dist_x / timediff
        ros_y = dist_y / timediff
        
        ros_x_list[i] = ros_x
        ros_y_list[i] = ros_y
    
    ros_x_mean = np.mean(ros_x_list)
    ros_y_mean = np.mean(ros_y_list)
    
    if output == "x_y":
        return(ros_x_mean, ros_y_mean)
    elif output == "ros_angle":
        ros_mean = math.sqrt(ros_x_mean**2 + ros_y_mean**2)
        angle_mean = get_angle_from_displacement(x = ros_x_mean, y = ros_y_mean)   
        return(ros_mean, angle_mean)




def get_ros_from_progress_array(progress_array, levels, output, max_interval = None):
    '''
    Parameters
    ----------
    progress_array : TYPE
        DESCRIPTION.
    levels : integer
        Indicates the size of the moving window, measured from the
        center pixel to the edge of the moving window
        (eg. level = 1 creates a 3 x 3 moving window).
        Note: levels directly influence the maximum possible ROS, for example
        with levels = 5 the maximum ROS that can be calculated is 5 pixels per frame
    output : string
        One of "ros_angle" or "x_y".
    max_interval : integer, optional
        the largest time difference between the pixel of interest
        (the central pixel) and the other pixels within the moving window.
        max_interval = None allows all time interavls no matter how large.
        The default is None.

    Returns
    -------
    ros_arr : TYPE
        DESCRIPTION.
    
    '''

    
    ros_arr = np.empty((progress_array.shape[0], progress_array.shape[1], 3))
    ros_arr[:] = np.nan
    
    
    for row in range(0, progress_array.shape[0]):
        
        print(f"Working on row {row}")
        
        for col in range(0, progress_array.shape[1]):
            
            # print(f"Working on col {col}")
            
            # if required subset is entirely within the array
            if row - levels >= 0 and col - levels >= 0:
                ar = progress_array[row - levels : row + levels + 1,
                                   col - levels : col + levels + 1]
            
            # if required subset is partially outside the array
            else:
                ar = np.empty((levels * 2 + 1, levels * 2 + 1))
                ar[:] = np.nan
                subset = progress_array[max(0, row - levels) : row + levels + 1,
                                   max(0, col - levels) : col + levels + 1]
                
                x = max(0, row - levels) - (row - levels)
                y = max(0, col - levels) - (col - levels)
                # insert subset into ar (array of nans)
                ar[x : x + subset.shape[0], y : y + subset.shape[1]] = subset
                
                
            mean_ros = get_mean_ros(ar, output, max_interval = max_interval)
            
            ros_arr[row, col, 0] = mean_ros[0]
            ros_arr[row, col, 1] = mean_ros[1]
      
    ros_arr[:, :, 2] = progress_array
    
    return ros_arr


def get_pixel_xsize_ysize(geotransform):
    """
    Note about geotransform format:
    [0] x-coordinate of the center of the upper left pixel
    [1] x-component of the pixel width (x-scale) (A)
    [2] y-component of the pixel width (y-skew) (D)
    [3] y-coordinate of the center of the upper left pixel (F)
    [4] x-component of the pixel height (x-skew) (B)
    [5] y-component of the pixel height (y-scale), typically negative (E)  
    """

    pixel_xsize = np.sqrt((geotransform[1]**2 + geotransform[2]**2))
    pixel_ysize = np.sqrt((geotransform[4]**2 + geotransform[5]**2))
    
    return(pixel_xsize, pixel_ysize)


def xy_array_ppf_to_ms(array, geotransform, fps):
    
    assert(array.shape[2] == 3)    
    array_ms = np.zeros(array.shape)
    
    pixel_xsize, pixel_ysize = get_pixel_xsize_ysize(geotransform)
        
    array_ms[:, :, 0] = array[:, :, 0] * pixel_xsize * fps
    array_ms[:, :, 1] = array[:, :, 1] * pixel_ysize * fps
    array_ms[:, :, 2] = array[:, :, 2]
    
    return array_ms



def xy_to_rosangle(array):
    assert(array.shape[2] == 3)    
    array_rosangle = np.zeros(array.shape)
    
    ros_x = array[:, :, 0]
    ros_y = array[:, :, 1]
    
    rate = np.sqrt(ros_x**2 + ros_y**2)
    angle = get_angle_from_displacement(ros_x, ros_y)
    
    array_rosangle[:, :, 0] = rate
    array_rosangle[:, :, 1] = angle
    array_rosangle[:, :, 2] = array[:, :, 2]
    
    return array_rosangle














