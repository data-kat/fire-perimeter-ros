# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:04:25 2020

@author: K Melnik


A collection of classes and functions for processing videos and the resulting
arrays in preparation for calculating the rate of fire spread
"""


import cv2
import math
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from osgeo import gdal
import affine
import fiona
import os





###############################################################################
####                   WORKING WITH IMAGES AND VIDEO                       ####
###############################################################################


class VideoExtractor:
    """ Opens a video file and extracts information and individual frames.
    
    Attributes
    ----------
    filename : str
        absolute or relative filepath including filename.
    frame_rate : int
        the frame rate (number of frames per second) of the input video.
    width : int
        the width of the video in pixels.
    height : int
        the height of the video in pixels.
    total_frames : int
        the total number of frames in the original video.
    
    """
    
    def __init__(self, filename):
        """ Initializes an instance of the class
        
        Parameters
        ----------
        filename : str
            absolute or relative filepath including filename.

        Returns
        -------
        None.
        
        """
        
        self.filename = filename
        self._cap = cv2.VideoCapture(self.filename)
        self.frame_rate = self.get_frame_rate(self._cap)
        
        self.width = int(self._cap.get(3))
        self.height = int(self._cap.get(4))
        self.total_frames = self._cap.get(7)
        
        self._cap.release()
    
    @staticmethod
    def get_frame_rate(capture):
        """ Retrieves the frame rate from the input video
        
        Parameters
        ----------
        capture : cv2.VideoCapture
            CV2 capture object from the input video

        Returns
        -------
        frame_rate : int
            the frame rate (number of frames per second) of the input video.

        """
        
        frame_rate = capture.get(5)
        return frame_rate
    
    def extract_frame(self, frame_index):
        """ Extracts one frame of the provided index from the video

        Parameters
        ----------
        frame_index : int
            The number of the frame that is to be extracted

        Returns
        -------
        frame : numpy.ndarray
            A 3D numpy array of shape (height, width, color_channels)
            containing one video frame
            
        """
        
        cap = cv2.VideoCapture(self.filename)        
        # Set the index of the frame you want to extract:
        cap.set(1, frame_index)        
        ret, frame = cap.read()        
        cap.release()        
        return frame
    
    def extract_frames(self, frame_indices):
        """ Extracts multiple frames from the input video based on frame indices

        Parameters
        ----------
        frame_indices : Union[list, tuple]
            A list or tuple containing one to many indices of frames to extarct

        Returns
        -------
        frame_list : list
            A list of 3D numpy arrays each containging one video frame
            
        """
        
        cap = cv2.VideoCapture(self.filename)  
        frame_list = []
        for frame_index in frame_indices:
            # Set the index of the frame you want to extract:
            cap.set(1, frame_index)        
            ret, frame = cap.read()     
            frame_list.append(frame)
        cap.release()        
        return frame_list
    
    def save_frames(self, destination_img_folder, desired_fps = None):
        """
        
        If the argument 'desired_fps' isn't passed in, the output framerate is
        set to the same value as the framerate of the input video'
        
        Parameters
        ----------
        destination_img_folder : str
            The absolute or relative destination path including filename
        desired_fps : int, optional
            The framerate of the output video. The default is None.

        Returns
        -------
        None.

        """
        
        cap = cv2.VideoCapture(self.filename)
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if desired_fps == None:
                desired_fps = self.frame_rate
            if frame_id % math.floor(self.frame_rate/desired_fps) == 0:
                filename = destination_img_folder + "/image_" +  str(int(frame_id)) + ".tif"
                cv2.imwrite(filename, frame)
        cap.release()
        print("Done!")
        
        
        
    def save_video_low_fps(self, new_fps = None, stop_at_frame = None):
        
        if new_fps is None:
            raise Exception("enter new fps (should be lower than original)")
        elif new_fps >= self.frame_rate:
            raise Exception("new fps should be lower than original")
        
        if stop_at_frame is None:
            stop_at_frame = self.total_frames
        
        cap = cv2.VideoCapture(self.filename)
        
        orig_path_no_ext = os.path.splitext(self.filename)[0]
        output_video_path = fr"{orig_path_no_ext}_{new_fps}_fps.mp4"
        
        output_video = cv2.VideoWriter(output_video_path, 0, new_fps, (self.width, self.height))
        print(f"created {output_video_path}")
        
        
        num_frames_to_merge = self.frame_rate/new_fps
        
        ids_to_extract_each_sec = [math.floor((i + 1) * num_frames_to_merge) - 1 for i in range(new_fps)]
        
        frames_to_merge = []
        frame_ids_to_merge = []
        
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if frame_id > stop_at_frame:
                break
                     
            first_frame_of_current_sec = int(math.floor((frame_id)/self.frame_rate) * self.frame_rate)
            
            ids_to_extract = [int(idx + first_frame_of_current_sec) for idx in ids_to_extract_each_sec]
            # print(f"frame_id: {frame_id}, first_frame_of_current_sec: {first_frame_of_current_sec}, ids_to_extract: {ids_to_extract}")
            
            for i, idx in enumerate(ids_to_extract):
                
                if i == 0:
                    prev_idx = first_frame_of_current_sec - 1
                else:
                    prev_idx = ids_to_extract[i-1]
                    
                # print(f"frame_id: {frame_id}, idx: {idx}, prev_idx: {prev_idx}")
                
                if prev_idx < frame_id < idx:
                    frames_to_merge.append(frame)
                    # print(f"frame {frame_id} appended")
                    frame_ids_to_merge.append(frame_id)
                    # print(f"frame {frame_id} appended; idx: {idx}, prev_idx: {prev_idx}")
                elif frame_id == idx:
                    frames_to_merge.append(frame)
                    # print(f"frame {frame_id} appended")
                    frame_ids_to_merge.append(frame_id)
                    # print(f"frame {frame_id} appended and all will be merged; idx: {idx}, prev_idx: {prev_idx}")
                    print(f"--Merging frames {str(frame_ids_to_merge).strip('[]')} and writing to file ({self.total_frames} total to process)")
                    average_frame = np.mean(frames_to_merge, axis=0).astype("uint8")                
                    frames_to_merge = [] # reset to empty tuple
                    frame_ids_to_merge = [] # reset to empty tuple
                    output_video.write(average_frame)
                else:
                    continue
            
        cap.release()
        output_video.release()
        print("Done!")


def rescale(image, scale, interpolation = None):
    
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST
    
    image_scaled = cv2.resize(
        image,
        (0,0),
        fx = scale,
        fy = scale,
        interpolation = interpolation)
    return image_scaled



def imshow_colorbar(array, interpolation = None):
    """ Displays an array with a colorbar using matplotlib
    
    Parameters
    ----------
    array : numpy.ndarray
        An image to be plotted/displayed
    interpolation : str, optional
        The interpolation to be used by matplotlib.pyplot.imshow function.
        Supported values are 'none', 'antialiased', 'nearest', 'bilinear',
        'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
        'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
        'sinc', 'lanczos'. The default is None. If interpolation is set to None,
        'none' is supplied to matplotlib.pyplot.imshow

    Returns
    -------
    None.

    """
    
    if interpolation is None:
        interpolation = 'none'
    
    fig, ax = plt.subplots()
    im = ax.imshow(array, interpolation = interpolation)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    
    fig.colorbar(im, cax = cax)    



def get_rainbow_tuples(n, saturation = 1, value = 0.8):
    """ Generates tuples of RGB color values indicating colors of the rainbow

    Parameters
    ----------
    n : int
        number of colors to generate.
    saturation : float, optional
        The intensity of the color between 0 and 1. High saturation colors are
        vibrant and pure. Low saturation colors are washed-out, whitish, and
        muted. The default is 1.
    value : float, optional
        The brightness (lightness/darkness) of the color between 0 and 1.
        low values are black, high values are bright (like shining a white
        light on a colored object, causing it to appear brighter.
        The default is 0.8.

    Returns
    -------
    rgb_tuples : list
        A list of RGB tuples of the specified length

    """
    
    hsv_tuples = [(x*1.0/n, saturation, value) for x in range(n)]
    rgb_tuples0 = [colorsys.hsv_to_rgb(hsv_tuple[0], hsv_tuple[1], hsv_tuple[2]) for hsv_tuple in hsv_tuples]
    # Multiply each number by 255:
    rgb_tuples = [[int(number*255) for number in rgb_tuple] for rgb_tuple in rgb_tuples0]
    return rgb_tuples


def bgr_to_hsv(image):
    """ Converts an image from the BGR format to the HSV format

    Parameters
    ----------
    image : numpy.ndarray
        The image in the BGR format to be converted. Opencv reads images into
        the BGR format

    Returns
    -------
    image_hsv : numpy.ndarray
        Output image in HSV format

    """
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image_hsv


def blur_dilate_erode(image, blur = None, iterations = None):
    """ Blurs, dilates and erodes a given image the required number of times

    Parameters
    ----------
    image : numpy.ndarray
        DESCRIPTION.
    blur : int, optional
        The size (in pixels) of the blur kernel. Must be positive and odd.
        The default is None.
    iterations : int, optional
        The number of iterations to run blur, dilate and erode algorithms.
        The default is None.

    Returns
    -------
    image_blurred_de : numpy.ndarray
        The output blurred, dilated and eroded image.

    """
    
    if blur is None:
        blur = 1
    
    if iterations is None:
        iterations = 1
    
    image_blurred = cv2.GaussianBlur(image, (blur, blur), 0)
    image_blurred_d = cv2.dilate(image_blurred, None, iterations = iterations)
    image_blurred_de = cv2.erode(image_blurred_d, None, iterations = iterations)
    
    return image_blurred_de  


class MaskAndCntFinder():
    """ Finds fire-affected areas in an image and makes a mask and contours
    
    Attributes
    ----------
    blur : int
        size of the blur kernel in pixels    
    bgr : image_bgr
        The input image in BGR color format
    range_list : range_list_hsv
        A list containing tuples of HSV color ranges
    mask_list : self.get_mask_list()
        A list of masks
    master_mask : self.get_master_mask()
        The main mask that encompasses all fire-afected areas
    master_cnts : self.get_cnts(self.master_mask)
        The main contours encompassing all fire affected areas
    cnts_list : self.get_cnts_list()
        The list of all contours
    
    """
    
    def __init__(self, image_bgr, range_list_hsv, blur = None):
        """ Initializes an instance of the class

        Parameters
        ----------
        image_bgr : numpy.ndarray
            The input image in BGR color format.
        range_list_hsv : list
            List of HSV color tuples indicating color threshold ranges.
        blur : int, optional
            Size of the blur kernel in pixels. The default is None.

        Returns
        -------
        None.

        """
        
        # Instantiate an object of ImageHsvProcessor class
        
        if blur is None:
            blur = 1
        
        self.blur = blur        
        self.bgr = image_bgr
        self.range_list = range_list_hsv
        self.mask_list = self.get_mask_list()
        self.master_mask = self.get_master_mask()
        self.master_cnts = self.get_cnts(self.master_mask)
        self.cnts_list = self.get_cnts_list()
    
    def get_one_mask(self, range_hsv):
        """ Retrives a binary mask from an image given one HSV range
        
        Before getting the mask, the image is blurred, dilated and eroded twice

        Parameters
        ----------
        range_hsv : list
            The desired minimums and maximums of hue, saturation and value.
            Format: [(min_hue, min_sat, min_val), (max_hue, max_sat, max_val)].
            Example: [(16, 0, 0), (163, 255, 67)]

        Returns
        -------
        mask : numpy.ndarray
            A 2D numpy array with 0's for pixels outside the mask, and 255's
            for pixels inside the mask.

        """
        
        image_hsv = bgr_to_hsv(self.bgr)
        
        mask = cv2.inRange(
            blur_dilate_erode(image_hsv, blur = self.blur, iterations = 2),
            range_hsv[0],
            range_hsv[1])
        return mask
    
    def get_mask_list(self):
        """ Produces a list of masks from the class instance
        
        Uses the input image and the the input HSV ranges

        Returns
        -------
        mask_list : list
            A list of masks (numpy arrays containing 0's and 255's).

        """
        
        mask_list = []
        
        for range_hsv in self.range_list:
            mask = self.get_one_mask(range_hsv)
            mask_list.append(mask)
        return mask_list

    def get_master_mask(self):
        """ Merges all masks to produce one main mask for the class instance

        Returns
        -------
        master_mask : numpy.ndarray
            A 2D numpy array with 0's for pixels outside the mask, and 255's
            for pixels inside the mask.

        """
        
        master_mask = self.mask_list[0]
        for i, mask in enumerate(self.mask_list):
            if i > 0:
                master_mask = cv2.bitwise_or(master_mask, mask)
        return master_mask
    
    @staticmethod
    def get_cnts(mask):
        """ Produces contours given a mask

        Parameters
        ----------
        mask : numpy.ndarray
            A 2D numpy array with 0's for pixels outside the mask, and 255's
            for pixels inside the mask.

        Returns
        -------
        cnts_clean : list
            List of numpy.ndarrays, where each array is a contour polygon.

        """
        # Find all contours
        cnts, hierarchy  = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Remove all contours that appear inside other contours (if hierarchy != -1)
        cnts_clean = []
        for j, cnt in enumerate(cnts):
            if hierarchy[0, j, 3] == -1:
                cnts_clean.append(cnt)
                
        return cnts_clean
            
            
    def get_cnts_list(self):
        """ Loops through the mask list to get a list of contours

        Returns
        -------
        cnts_list : list
            List of lists of numpy.ndarrays, where each array is a contour
            polygon.

        """
        
        cnts_list = []
        for i, mask in enumerate(self.mask_list):
            cnts = self.get_cnts(mask)            
            cnts_list.append(cnts)
        return cnts_list  
    
    
    def draw_cnts(self, merged = None, display = None, return_image = None,
                  thickness = None):
        """ Draws the contours on the input image

        Parameters
        ----------
        merged : bool, optional
            Merge all the contours fro mthe different color ranges into one
            set of contours, or leave them unmerged with each color range
            having a separate set of contours? The default is None.
        display : bool, optional
            Show the output image? The default is None.
        return_image : bool, optional
            Return the output image? The default is None.
        thickness : int, optional
            The thickness of the contour outline. -1 means the contour polygon
            will be filled in. The default is None.

        Returns
        -------
        img_with_conts : numpy.ndarray
            The resulting image where contours have been overlaid over the
            original input image.

        """
        
        if merged is None:
            merged = False
        
        if display is None:
            display = True
            
        if return_image is None:
            return_image = False
            
        if thickness is None:
            thickness = 1
        
        # Generate some colors for coloring and displaying the contours
        colors_bgr = get_rainbow_tuples(len(self.cnts_list), value = 1)
        
        img_with_conts = self.bgr.copy()
        
        if merged == False:
            # Draw contours from each hsv range separately        
                
            for i, cnts in enumerate(self.cnts_list):
                cv2.drawContours(img_with_conts, cnts, -1, colors_bgr[i], thickness)
                
        elif merged == True:
            # Draw only the main contours outlining all the hsv ranges combined
            cv2.drawContours(img_with_conts, self.master_cnts, -1, colors_bgr[0], thickness)
        
        if display == True:
            cv2.imshow('img_with_conts', img_with_conts)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if return_image == True:
            return img_with_conts
    

#############################################################
# Function to determine the needed hsv_ranges using trackbars

# Preparing Track Bars
# Defining empty function
def do_nothing(x):
    pass


def create_hsv_trackbars(track_variables):
    
    # Giving name to the window with Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    
    for var in track_variables:
        cv2.createTrackbar(var, 'Track Bars', 0, 255, do_nothing)

def find_ideal_range_video(video_fn, num_ranges = None):
    
    if num_ranges is None:
        num_ranges = 1
    elif num_ranges > 6:
        raise Exception("Max number of ranges allowed is 6")
    
    
    video_object = VideoExtractor(video_fn)
    
    # Open the video
    cap = cv2.VideoCapture(video_fn)
    
    # Defining Track Bars for convenient process of choosing colours
    
    track_variables_list = []
    letters = ["A", "B", "C", "D", "E", "F"]
    
    for i in range(num_ranges):
        letter = letters[i]
        
        track_variables = (f"{letter}-min_hue",
                           f"{letter}-min_sat",
                           f"{letter}-min_val",
                           f"{letter}-max_hue",
                           f"{letter}-max_sat",
                           f"{letter}-max_val")
        track_variables_list.append(track_variables)
    # Flatten the list
    track_variables = [item for sublist in track_variables_list for item in sublist]
        
    create_hsv_trackbars(track_variables)
    
    # Trackbar for scrubbing through footage:
    cv2.createTrackbar("Frame number", 'Track Bars', 0, int(video_object.total_frames-1), do_nothing)
    
    # Trackbar for blur:
    cv2.createTrackbar("Blur", 'Track Bars', 1, 21, do_nothing)
    
    
    # Defining loop for choosing right Colours for the Mask
    while True:
        
        # Get the frame number of the frame you want to view:
        current_frame_id = cv2.getTrackbarPos("Frame number", 'Track Bars')
        # Set the index of the frame you want to view:
        cap.set(1, current_frame_id)        
        ret, image_bgr = cap.read()
        
        blur = cv2.getTrackbarPos("Blur", 'Track Bars')
        if blur % 2 == 0:
            blur = blur + 1
        
        # Getting the current values from the track bars:        
        track_values = dict()
        for var in track_variables:
            track_values[var] = cv2.getTrackbarPos(var, 'Track Bars')
            
        # Defining lower bounds and upper bounds for thresholding
        
        range_list = []
        for i in range(num_ranges):
            letter = letters[i]
            
            hsv_min = (track_values[f"{letter}-min_hue"],
                       track_values[f"{letter}-min_sat"],
                       track_values[f"{letter}-min_val"])
            
            hsv_max = (track_values[f"{letter}-max_hue"],
                       track_values[f"{letter}-max_sat"],
                       track_values[f"{letter}-max_val"])
            range_list.append([hsv_min, hsv_max])
       
        
        mask_cnt_object = MaskAndCntFinder(image_bgr, range_list, blur = blur)
        
        cnts_img1 = mask_cnt_object.draw_cnts(display = False,
                                             return_image = True,
                                             thickness = 1)
        
        cnts_img2 = mask_cnt_object.draw_cnts(display = False,
                                             return_image = True,
                                             thickness = -1)
        
        cv2.namedWindow('cnts_img1', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("cnts_img1", 640, 480)
        cv2.imshow('cnts_img1', rescale(cnts_img1, 0.5, interpolation = cv2.INTER_CUBIC))
        
        cv2.namedWindow('cnts_img2', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("cnts_img2", 640, 480)
        cv2.imshow('cnts_img2', rescale(cnts_img2, 0.5, interpolation = cv2.INTER_CUBIC))
        
    
        # Breaking the loop if 'Escape' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            break
    
    
    # Destroying all opened windows
    cv2.destroyAllWindows()
    
    # Closing the video
    cap.release()    
    
    return track_values



def check_ranges_on_video(video_fn, range_list):
    
    video_object = VideoExtractor(video_fn)
    
    # Open the video
    cap = cv2.VideoCapture(video_fn)
    
    # Defining Track Bars for convenient process of choosing colours
    
    # Giving name to the window with Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    
     # Trackbar for scrubbing through footage:
    cv2.createTrackbar("Frame number", 'Track Bars', 0, int(video_object.total_frames-1), do_nothing)
    
    # Trackbar for blur:
    cv2.createTrackbar("Blur", 'Track Bars', 1, 21, do_nothing)
    
    
    # Defining loop for choosing right Colours for the Mask
    while True:
        
        # Get the frame number of the frame you want to view:
        current_frame_id = cv2.getTrackbarPos("Frame number", 'Track Bars')
        # Set the index of the frame you want to view:
        cap.set(1, current_frame_id)        
        ret, image_bgr = cap.read()
        
        blur = cv2.getTrackbarPos("Blur", 'Track Bars')
        if blur <= 0:
            blur = 1
        elif blur % 2 == 0:
            blur = blur + 1
        
        mask_cnt_object = MaskAndCntFinder(image_bgr, range_list, blur = blur)
        
        cnts_img1 = mask_cnt_object.draw_cnts(display = False,
                                             return_image = True,
                                             thickness = 1)
        
        cnts_img2 = mask_cnt_object.draw_cnts(display = False,
                                             return_image = True,
                                             thickness = -1)
        
        cv2.namedWindow('cnts_img1', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("cnts_img1", 640, 480)
        cv2.imshow('cnts_img1', rescale(cnts_img1, 0.5, interpolation = cv2.INTER_CUBIC))
        
        cv2.namedWindow('cnts_img2', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("cnts_img2", 640, 480)
        cv2.imshow('cnts_img2', rescale(cnts_img2, 0.5, interpolation = cv2.INTER_CUBIC))
        
    
        # Breaking the loop if 'Escape' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            break
    
    
    # Destroying all opened windows
    cv2.destroyAllWindows()
    
    # Closing the video
    cap.release()





##################





def get_progress_mask_and_cnts(image, range_list, burnt_pixels, blur = None):
    """ Produce a mask of additonal pixels  burnt compared to the previous mask

    Parameters
    ----------
    image : numpy.ndarray
        An input BGR image (3D numpy array) obtained from the source video
        using opencv.
    range_list : list
        A list of lists containing HSV ranges/thresholds of interest.
    burnt_pixels : numpy.ndarray
        A mask of pixels locations that rae considered fire-affected/"burnt".

    Returns
    -------
    mask_dif : numpy.ndarray
        Mask resulting from subtracting the "burnt pixels" mask from pixels
        that were newly burnt in the inout frame.
    cnts_dif : list
        The list of contour polygons (each contour is an array) outlining the
        newly burnt areas, i.e. outlining the mask of newly burnt pixels .
    burnt_pixels_new : numpy.ndarray
        The new cumulative mask showing all burnt pixels up to and including
        the current frame.

    """
        
    mask = MaskAndCntFinder(image, range_list, blur = blur).master_mask
    
    #Find the difference between the new mask and the burnt_pixels master mask
    mask_dif = subtract_masks(mask, burnt_pixels)
    
    # Find the contours in the difference-mask
    cnts_dif = MaskAndCntFinder.get_cnts(mask_dif)
    
    # Add the current contour analyzed to burnt pixels array:
    burnt_pixels_new = add_masks(mask, burnt_pixels)
    
    return mask_dif, cnts_dif, burnt_pixels_new



def add_masks(*masks):
    """ Adds multiple masks together

    Parameters
    ----------
    *masks : numpy.ndarray
        Contains 0's and 255's only. 0 means absence, 255 means presence.

    Returns
    -------
    masks_sum : numpy.ndarray
        The sum of all the input masks.

    """
    
    masks_sum = masks[0]
    
    if len((masks)) > 1:
        for mask in masks[1:]:
            masks_sum = cv2.bitwise_or(masks_sum, mask)
    
    return masks_sum


def subtract_masks(mask1, mask2):
    """ Substracts two masks and output the result of mask1 - mask2

    Parameters
    ----------
    mask1 : numpy.ndarray
        Mask to substracts from. Contains 0's and 255's only.
    mask2 : numpy.ndarray
        Maks to be subtracted. Contains 0's and 255's only.

    Returns
    -------
    mask_difference : numpy.ndarray
        The resulting mask representing the difference between the two masks.

    """
    mask_difference = cv2.subtract(mask1, mask2)
    return mask_difference



def add_mask_to_array(mask, mask_id, array):
    """

    Parameters
    ----------
    mask : numpy.ndarray
        DESCRIPTION.
    mask_id : int
        DESCRIPTION.
    array : numpy.ndarray
        DESCRIPTION.

    Returns
    -------
    updated_array : TYPE
        DESCRIPTION.

    """
    mask_with_id = np.array(mask/255*(mask_id + 1), dtype="uint16")
    updated_array = add_masks(mask_with_id, array)
    return updated_array



def get_progress_array_and_cnts_list(video_filename, range_list,
                                     desired_fps = None, max_frame = None,
                                     blur = None):
    
    cap = cv2.VideoCapture(video_filename)
    ret, first_frame = cap.read()
    frame_rate = cap.get(5)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frame is None:
        max_frame = total_frames
        
    if desired_fps is None:
        desired_fps = frame_rate
    
    # Initialize objects to be altered in the while-loop
    height = first_frame.shape[0]
    width = first_frame.shape[1]
    
    burnt_pixels = np.zeros((height, width), dtype="uint8")
    prog_arr = np.zeros((height, width), dtype="uint16")
    prog_cnts_list = []
       
    while(cap.isOpened()):
        frame_id = cap.get(1) #current frame number
        
        if frame_id > max_frame:
            break
        
        print(f"processing frame {int(frame_id)} of {total_frames}")
        
        ret, frame = cap.read()
        if (ret != True):
            break
        if frame_id % math.floor(frame_rate/desired_fps) == 0:
            mask_dif, cnts_dif, burnt_pixels_new = get_progress_mask_and_cnts(
                frame,
                range_list,
                burnt_pixels,
                blur = blur)
            
            # Update summary progression array, prog_cnts_list and burnt_pixels:
            prog_arr = add_mask_to_array(mask_dif,
                                         frame_id + 1, # add an extra 1 tomake up for making all zeros into np.nan at the end
                                         prog_arr)
            prog_cnts_list.append(cnts_dif)
            burnt_pixels = burnt_pixels_new
            
            # view(burnt_pixels)
    
    prog_arr = prog_arr.astype("float")
    prog_arr[prog_arr == 0] = np.nan
    
    prog_arr = prog_arr - 1 # now that we made all zeros into np.nans, can reom the 1 we added earlier
    
    return prog_arr, prog_cnts_list

    cap.release()
    print("Done!")   




###############################################################################
####                         GEAOSPATIAL ANALYSIS                          ####
###############################################################################

def read_lines(links_filename):
    with open(links_filename, "r") as file:
        lines = file.read().split('\n')
        return lines


def line_to_numbers(line):    
    # Convert the line to a list of numbers    
    line_str = line.split("\t")
    line_float = [float(i) for i in line_str] # convert each number fr. str to float
    return line_float


# extract coordinates from the ArcMap-generated links text file line by line.
#   Format : X_Map, Y_Map, elevation(0), X_Source, -Y_Source
def gcp_from_number_list(number_list, scale = None):
    
    if scale is None:
        scale = 1
    
    gcp = gdal.GCP(number_list[2], number_list[3], 0, number_list[0] * scale, -number_list[1] * scale)
    return gcp



def gcps_from_links_file(links_filename, scale = None):
    raw_lines = read_lines(links_filename)
    
    gcps = []
    
    # If line is not blank, covert it to gcp
    # raw_line = raw_lines[0]
    for raw_line in raw_lines:
        if raw_line != "":
            number_list = line_to_numbers(raw_line)
            gcp = gcp_from_number_list(number_list, scale = scale)
            gcps.append(gcp)
        
    return gcps



def geotransform_from_gcps(gcps):
    geotransform = gdal.GCPsToGeoTransform(gcps)
    return geotransform



def get_coords_from_shapefile(shapefile_name, geometry = None):
    with fiona.open(shapefile_name) as shapefile:
        
        if geometry == "line":
            proj_coordlist = shapefile[0]['geometry']['coordinates']
        elif geometry == "polygon":
            proj_coordlist = shapefile[0]['geometry']['coordinates'][0]
        else:
            raise Exception(f"geometry is {geometry}, plese set to 'line' or 'polygon'")

        return proj_coordlist


def proj_coord_to_pixel_coord(proj_coord, image_geotransform):
    affine_transform = affine.Affine.from_gdal(*image_geotransform) # asterisk is important
    reverse_affine_transform = ~affine_transform
    pixel_coord = reverse_affine_transform * proj_coord
    return pixel_coord


def proj_coord_to_pixel_coord_list(proj_coord_list, image_geotransform):
    pixel_coord_list = []
    for proj_coord in proj_coord_list:
        pixel_coord = proj_coord_to_pixel_coord(proj_coord, image_geotransform)
        pixel_coord_list.append(pixel_coord)
    return pixel_coord_list


def pixel_coords_from_shapefile(shapefile_name, geotransform, geometry = None):
       
    proj_coordlist = get_coords_from_shapefile(shapefile_name, geometry = geometry)
    pixel_coord_list = proj_coord_to_pixel_coord_list(proj_coordlist, geotransform)
    
    return pixel_coord_list



def polygon_mask_from_shapefile(shapefile_name, image, geotransform):
    
    poly_pixcoords = pixel_coords_from_shapefile(
        shapefile_name,
        geotransform,
        geometry = "polygon"
        )
    
    # Get an array containing all the points inside the polygon:
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    
    # Find all pixels inside polygon
    polygon_path = Path(poly_pixcoords) # make a polygon
    polygon_grid = polygon_path.contains_points(points)
    # create a mask with points inside a polygon:    
    polygon_mask = polygon_grid.reshape(image.shape[0], image.shape[1])
    
    return polygon_mask











