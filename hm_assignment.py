import cv2 as cv
import os
from math import sqrt, pi
# from matplotlib.pyplot import contour
import numpy as np
from os import listdir
from cv2 import vconcat, imshow
from scipy.spatial import distance as dist

# Path for images
path = "bottle_crate_images/"
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

    count = 1  # Counter for component naming
    last_marked = None  # Last marked component as a bottle
    # current_component = None  # The current component being analysed
    for i in range(1, num_of_labels):
        # Extract the statistics of each component
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        current_component = centroids[i]
        if area > 120 and area < 250:
            # and w <= 45 and w >= 35 unused metric for comparing components
            if h >= 30 and h <= 55 and w >= 30 and w <= 55 and abs(w - h) < 5:
                if last_marked is None:  # Identifying first component

                    cv.rectangle(img_filtered, (x, y),
                                 (x + w, y + h), (0, 255, 0), 1)
                    cv.putText(img_filtered, str(count), org=(int(current_component[0]), int(current_component[1])),
                               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_8)
                    print(
                        f"centroids:{current_component}\tcount:{count}\twidth: {w}\theight:{h}")
                else:
                    # Check the distance between the centre of the current component and the previous component
                    distance = dist.euclidean(last_marked, current_component)
                    if distance < 90:
                        continue
                    cv.rectangle(img_filtered, (x, y),
                                 (x + w, y + h), (0, 255, 0), 1)
                    cv.putText(img_filtered, str(count), org=(int(current_component[0]), int(current_component[1])),
                               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_8)
                    print(
                        f"count:{count}\tcentroids:{current_component}\twidth: {w}\theight:{h}\tdistance:{distance:.2f}")
                last_marked = current_component  # Last marked component as a bottle
                count = count + 1

    # Draw rectangle around component
    imshow("Components", img_filtered)
    cv.waitKey(0)


# Loop through images and select each image to find circles
for index, img in enumerate(img_dir):
    img = cv.imread(path+img_dir[index], 0)
    edges = cv.Canny(img, 80, 200, apertureSize=3)
    edges = cv.dilate(edges, kernel=np.ones(
        (1, 1), np.uint8))
    component_analysis(edges, img)
    # exit()
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


Modifications to make
1. Check a particular height that fits most of the components.
    How do I identify the first component that is a bottle.
2. Figure out the components that have vastly uneven widht and height and remove them.
3. Find a way to remove all non-bottle components that are identified. One way could be the 
    euclidean distance between the centre points of the  individual components identified.
    If the centre points are within a distance closer to each other, ignore the current component
"""
