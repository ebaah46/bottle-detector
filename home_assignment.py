
"""
Authors: Emmanuel Baah & Jian Zhang

Procedure followed
1. Load the image.
2. Apply filters to remove noise.
3. Run hough circles algorithm.
4. Test values for minimum distance, minimum radius, maximum radius that satisfies desired object.
5. Draw circle around obtained bottle component.
6. Label the circle with count value.

The main resources used for this project are

OpenCV - Documentation
https://docs.opencv.org/4.x/

Detecting circles in images using opencv and Hough Circles
https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/ 


"""

import argparse
import cv2 as cv
import numpy as np
from cv2 import imwrite

# Hough Circles approach


def hough_cirlces(image, original_image, name):
    """
    Function the implements the hough transform algo for bottle detection 

    Args:
        image (numpy array): grayscale image after filtering is done
        original_image (numpy array): copy of original image
        name (string): image file name

    Returns:
        tuple: returns number of bottles and generated image filename
    """
    min_dist = 30  # minimum distance between the centres of the detexted circles
    param_1 = 32  # higher threshold
    param_2 = 34   # accumulator threshold
    min_radius = 10  # minimum radius of the circle
    max_radius = 36  # maximum radius of the circle
    circle_count = 0
    # Run hough circles algorithm
    circles = cv.HoughCircles(
        image, cv.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=param_1, param2=param_2, minRadius=min_radius, maxRadius=max_radius)
    circles = np.uint16(np.around(circles))
    for index, circle in enumerate(circles[0, :]):
        # Draw circle around the bottle
        cv.circle(original_image, (circle[0],
                  circle[1]), circle[2], color=(0, 255, 255), thickness=3)
        # Place bottle count text inside the circle
        cv.putText(original_image, str(index+1), (circle[0], circle[1]), fontFace=cv.FONT_HERSHEY_COMPLEX,
                   fontScale=0.4, color=(206, 209, 0), thickness=2, lineType=cv.LINE_8)
        circle_count += 1  # increment circle count
    # Form new name for saved image
    name = f"{name}_{circle_count}_bottles.png"
    # Save new image file with identified circles
    imwrite(name, original_image)

    return (circle_count, name)


if "__main__" == __name__:
    agp = argparse.ArgumentParser()
    # Handle command line arguments
    agp.add_argument("-i", "--image",  help="provide a path to an image")
    args = vars(agp.parse_args())
    # Check if image is provided.
    # If not, end the program
    if "image" in args.keys():
        # read image into memory
        try:
            img = cv.imread(args["image"], cv.IMREAD_COLOR)
            # Make a copy of the actual image
            copy = img.copy()
            # Filter image until noise is considerably removed
            # apply median filter with a kernel size of 5
            img = cv.medianBlur(img, 5)
            # apply median filter with a kernel size of 7
            img = cv.medianBlur(img, 7)
            # apply bilateral filter with a filter size of 9
            img = cv.bilateralFilter(img, 9, 114, 114)
            # Run hough transform algorithm
            bottles, file = hough_cirlces(cv.cvtColor(
                img, cv.COLOR_BGR2GRAY), copy, f"{args['image']}")
            print(f"{bottles} found. Confirmation image is {file}")
        except:
            print("Failed to process image")
    else:
        print("No image provided")
