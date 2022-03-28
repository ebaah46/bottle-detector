from curses import raw
import cv2 as cv
import numpy as np
from os import listdir
from cv2 import imshow
from scipy.spatial import distance as dist


# Component analysis function
def component_analysis(image, raw_image):
    # component analysis with stats
    (num_of_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        image, 8, cv.CV_32S)
    # convert raw_image to color image to be displayed
    img_filtered = cv.cvtColor(raw_image, cv.COLOR_GRAY2BGR)

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
        # if area > 120 and area < 250:  # Discard all components outside the desired area values of the bottles
        if h >= 30 and h <= 55 and w >= 30 and w <= 55 and abs(w - h) < 5:
            if last_marked is None:  # Identifying first component
                # Draw a rectangle around the component
                cv.rectangle(img_filtered, (x, y),
                             (x + w, y + h), (0, 255, 0), 1)
                cv.putText(img_filtered, str(count), org=(int(current_component[0]), int(current_component[1])),
                           fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_8)
                print(
                    f"count:{count}\tcentroids:{current_component}\twidth: {w}\theight:{h}")
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

    imshow("Components", img_filtered)
    cv.waitKey(0)


# Hough Circles approach
def hough_cirlces(image, raw_image):
    cv.GaussianBlur(raw_image, (9, 9), 2, raw_image, 2)
    circles = cv.HoughCircles(
        raw_image, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=0.9, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    img_filtered = cv.cvtColor(raw_image, cv.COLOR_GRAY2BGR)
    print(len(circles[0]))
    for circle in circles[0, :]:
        cv.circle(img_filtered, (circle[0],
                  circle[1]), circle[2], color=(0, 255, 0), thickness=2)
        # print(circle)
    imshow(f"Image {index+1} v Image manipulated", img_filtered)
    cv.waitKey(0)


# Contours approach
def contours(image, raw_image):
    #  Draw contour around the crate
    contours, _ = cv.findContours(
        image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Convert image to color image
    raw_image = cv.cvtColor(raw_image, cv.COLOR_GRAY2BGR)
    # Last marked circle
    last_marked = None
    # Discovered circles
    circles = []
    count = 1
    for _, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        if h >= 30 and h <= 100 and w >= 30 and w <= 100:
            # compute center
            M = cv.moments(cnt)
            center = (int((M["m10"] / M["m00"]) + 1e-5),
                      int((M["m01"] / M["m00"]) + 1e-5))

            if last_marked is None:
                cv.rectangle(raw_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv.putText(raw_image, str(count), org=(int(center[0]), int(center[1])),
                           fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_8)
                last_marked = center
                count += 1
            # Save components of interest
            else:
                dst = dist.euclidean(last_marked, center)
                if dst > 95:
                    cv.rectangle(raw_image, (x, y),
                                 (x+w, y+h), (0, 0, 255), 0)
                    cv.putText(raw_image, str(count), org=(int(center[0]), int(center[1])),
                               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_8)
                    last_marked = center
                    count += 1
    imshow("Image with drawn contours", raw_image)
    cv.waitKey(0)


if "__main__" == __name__:

    # Path for images
    path = "bottle_crate_images/"
    # Load images directory
    img_dir = listdir(path)

    # Loop through images and select each image to find circles
    for index, img in enumerate(img_dir):
        # read image into memory
        img = cv.imread(path+img_dir[index], 0)
        # Applying a filter
        img = cv.bilateralFilter(img, 256, 85, 85, cv.BORDER_CONSTANT)
        # Detect edges in the image
        edges = cv.Canny(img, 80, 255, apertureSize=3)
        # morphological dilation
        # edges = cv.dilate(edges, kernel=np.ones((1, 1), np.uint8))
        imshow("Edges", edges)
        cv.waitKey(0)
        # component analysis on list of edges
        # component_analysis(edges, img)

        # Perform hough transform
        # hough_cirlces(edges, img)
        # Perform contour detection
        # contours(edges, img)
        continue
        # exit()


# TODO
# Algorithm for counting bottles in the crate


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
