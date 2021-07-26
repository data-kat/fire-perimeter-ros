# VFPT - Visual Fire Perimeter Tracking

Welcome to Visual Fire Perimeter Tracking. This package is designed to process a stabilized overhead visible-spectrum video of a spreading fire and output a 3-dimensional array of the same width and height as the video, with each pixel containing the rate of fire spread, the direction of fire spread, and the frame number at which the firefront first reached the given pixel.

Current Version: 0.1


## 1. Introduction

The package uses python and opencv to detect the location of the firefront in crop stubble using HSV color thresholding, and determine the rate and direction of spread at each pixel in the field of view based on the movement of the firefront when it reached each location.
In order to obtain the resulting array of fire spread vectors, the following operations need to be performed:

- color thresholding to identify flaming and burnt ('fire-affected') areas in each frame of the video
- constructing a fire progressing array by combining the fire-affected areas from all frames into one array
- analyze the fire progression array to calculate the rate and direction of fire spread


## 2. Segment out fire-affected area

Before we can measure the rate of spread, it is necessary to segment out fire-affected areas from unburnt areas on the RGB image. Because visible-spectrum images do not store any temperature information, we have to rely on color to accomplish this. Flaming and burnt areas have a distinctive color associated with them (orange and black, respectively) and are easier differentiated fro mthe rest of the image in the HSV (hue, saturation, value) color model compared to the RGB (red, green, blue) color model. The opencv library allows for easy easy conversion of the input RGB image to HSV.

The base.find_ideal_range_video function converts the image to the HSV format, and allows the user to manually select the hue, saturation and value ranges that represent the fire-affected areas in the video while minimizing the amount of false-positive identifications. The user can select how many ranges they want to be able to set at a time. Use the frame number scrollbar to scrub through the whole video to ensure the selected colors work for the entire video. The resulting ranges will be printed to the console - copy and paste the desired ones into your code to hardcode them.

- In the presented workflow when a pixel is registered as fire-affected it retains this status for the rest of the video. As a result, false-positived identifications are much more likely to skew the results than false-negatives, and should be minimized when possible. 

- The base.check_ranges_on_video function can be used to check that the entered ranges are accurate enough. Use the frame number scrollbar to scrub through the entire video


## 3. Fire progression array

This function analyzes each frame of the video and records the outlines of areas that became "fire-affected" at every given frame. The output is reported as an array with the timestamp indicating the time when each pixel became fire-affected, and it also returns a list of cv2 contours. Note: this will take a while to run.


## 4. Calculate ROS vectors

The rate of fire spread can be canculated for each pixel of the fire progress array by evaluating the time at which the nearby pixels became fire-affected, and the pixelwise distance to these nearby pixels. The output of this calculation is in pixels per frame, and needs to be converted to meters per second by introducing geospatial infomation about the stabilized input video. Because generally the width of the pixels in  meters are sliughtly different from the height of the pixels in meters, it is best to set the outout of this function to "x_y" (the x-axis ROS and the y-axis ROS), although "ros_angle" outout is also possible.


In order to translate velocity vectors from pixels per frame to meters per second, and correct their direction to reflect north-up orientation, we need to georeference our image and obtain a geotransform - a 6-number string that contains information about the pixel width, height and rotation, as well as the real-world coordinates of the top-left corner of the image. In order to achieve this, export the first frame of the video and georeference it in external software to obtain ground-control points - a list of locations with matched pixel-wise and geographic coordinates (I used ArcMap for this, but there are also open-source options such as QGIS).


The resulting ground control points were saved in links_plot1.txt. Read the resulting ground control points file and obtain a geotransform from these gcps:

    gcps_p1 = base.gcps_from_links_file(links_p1_filename)
    geotransform_p1 = base.geotransform_from_gcps(gcps_p1)

To convert ROS from pixels per frame to meters per second it is also necessary to know how many frames were recorded during each second. We can obtain this information from the input video by extracting its frame rate. Once we have the geotransform and the frame rate, velocities in pixels per frame can be converted to meters per second using the ros.xy_array_ppf_to_ms function. The last step for obtaining geographically true velocities is to adjust the vectors so that they reflect values with respect to 0 degrees being North rather than the top of the image. 

- Note: there are a lot of artefacts around the edges where the fire segmentation picked up previously burnt areas that were not involved in the current burn. Can remove these by clipping all regions that are outside the experimental plot to the shapefile that outline sthe extent of the research plot.















