import cv2 as cv
import os
from math import sqrt, pi
from matplotlib.pyplot import contour
import numpy as np
from os import listdir
from cv2 import vconcat, imshow


# Path for images
path = "/home/beks/School/MV/Home Assignment/bottle_crate_images/"
# Load images directory
img_dir = listdir(path)


# Component analysis function
def component_analysis(image, raw_image):
    # component analysis with stats
    (num_of_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        image, 8, cv.CV_32S)
    # make copy of provided image
    # img = image.copy()
    img_filtered = cv.cvtColor(raw_image, cv.COLOR_GRAY2BGR)
    # print(labels)
    # Counter for component naming
    count = 1
    for i in range(1, num_of_labels):
        # Extract the statistics of each component
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        cX, cY = centroids[i]
        if area > 120 and area < 200 and w <= 39 and w >= 35:
            # Check radius size for all circles
            radius = sqrt(area / pi)
            # print(f"Radius for circle with centre:{cX,cY} is:{radius}")
            # if (radius <= 6.59 and radius >= 6.30):
            cv.rectangle(img_filtered, (x, y),
                         (x + w, y + h), (0, 255, 0), 1)
            cv.putText(img_filtered, str(count), org=(int(cX), int(cY)),
                       fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv.LINE_8)
            count = count + 1
            print(f"width: {w}")
    # Draw rectangle around component
    cv.imshow("Components", img_filtered)
    cv.waitKey(0)


# Loop through images and select each image to find circles
for index, img in enumerate(img_dir):
    img = cv.imread(path+img_dir[index], 0)
    edges = cv.Canny(img, 80, 200, apertureSize=3)
    edges = cv.dilate(edges, kernel=np.ones(
        (1, 1), np.uint8))
    component_analysis(edges, img)
    continue


# TODO
# Algorithm for counting bottles in the crate

# Contours approach
""" 
    #  Draw contour around the crate
    contours, _ = cv.findContours(
        eroded_opened, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(f"No. of contours: {len(contours)}")
    print(f"contour 2: {contours[1]}")
    for contour in contours:
        approx = cv.approxPolyDP(
            contour, 0.1 * cv.arcLength(contour, True), True)
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.drawContours(img_bgr, contour, -1, (0, 255, 0), 3)

"""

# Hough Circles approach
""" 
    circles = cv.HoughCircles(
        edges, cv.HOUGH_GRADIENT, dp=1, minDist=80, param1=200, param2=0.95, minRadius=18, maxRadius=20)

    circles = np.uint16(np.around(circles))
    print(len(circles[0]))
    for circle in circles[0, :]:
        cv.circle(img_filtered, (circle[0],
                  circle[1]), circle[2], color=(0, 255, 0), thickness=2)
        # print(circle)
    img_combined = vconcat(
        [img_bgr, img_filtered])
    cv.imshow(f"Image {index+1} v Image manipulated", img_combined)
    cv.waitKey(0)

"""

# Connected Components approach
"""
1. Load all images.
2. Convert to binary image.
3. Perform canny edge detection on binary image.
4. Find all connected components in the image.
5. Filter through component properties for desired components.
6. Label the components in each image.
"""
