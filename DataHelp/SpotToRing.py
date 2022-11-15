import numpy as np
import skimage
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
from Utils import *

# TODO: Incorporate pixel to 1/nm scale for radii

# Take all zone axes of simulated TEM image and combine into single image
#   using a logical or
def CombineSimZoneAxes(dir):
    imageHeight, imageWidth = GetImageSize(f"{dir}/{os.listdir(dir)[0]}")
    combinedImage = np.zeros((imageHeight, imageWidth), dtype=int)
    for image in os.listdir(dir):
        try:
            image = skimage.io.imread(f"{dir}/{image}", as_gray=True)
            np.logical_or(image, combinedImage, out=combinedImage, where=[1])
        except ValueError:
            print(f"Skipping {image}")
    
    return combinedImage

# Assumes image is centered
def ShowRingImageFromSpots(image, rots=5):    
    for i in range(rots):
        rotImage = skimage.transform.rotate(image, i)
        np.logical_or(rotImage, image, out=image, where=[1])
    
    plt.imshow(image, cmap="gray")
    plt.show()

# Centers an image based on position of dots
def CenterImage(image, showTranslate=False):

    # Find center point
    height, width = GetImageSize(image, path=False)
    center = (width/2, height/2)

    # Find all points in image
    rows, cols = np.where(image != 0)

    # Find centroid and calculate appropriate transformation
    xAvg = np.average(cols)
    yAvg = np.average(rows)

    xTrans = xAvg - center[0]
    yTrans = yAvg - center[1]

    # Translate image to center
    image = Translate(image, xTrans, yTrans)

    # Display new image with calculated point center (red) and center of image (blue)
    if showTranslate:
        plt.imshow(image, cmap="gray")
        plt.scatter(xAvg, yAvg,s=50, c="r")
        plt.scatter(center[0], center[1], s=50, c="b")
        plt.show()
    
    # Re-find all the points in the image
    rows, cols = np.where(image != 0)
    points = list(zip(cols,rows))

    # Find point closest to center
    # FIXME: closest point position not found correctly
    distances = []
    pointDistance = lambda p1,p2: math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
    for point in points:
        distances.append(pointDistance(point, center))
    minDist = min(distances)
    minDistIndex = distances.index(minDist)
    centralPoint = points[minDistIndex]

    # Find x and y distance from central point to center of image
    xTrans = centralPoint[0] - center[0]
    yTrans = centralPoint[1] - center[1]

    image = Translate(image, xTrans, yTrans)

    # Display new image with central point (red) and center of image (blue)
    if showTranslate:
        plt.imshow(image, cmap="gray")
        plt.scatter(centralPoint[0], centralPoint[1], s=50, c="r")
        plt.scatter(center[0], center[1], s=50, c="b")
        plt.show()

    return image

# Calculate radius from central most point
def CalculateRadii(image):
    # Catalog all the points in the image
    rows, cols = np.where(image != 0)
    points = list(zip(cols,rows))

    # Find center point
    height, width = GetImageSize(image, path=False)
    center = (width/2, height/2)

    # Find point closest to center
    distances = []
    pointDistance = lambda p1,p2: math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
    for point in points:
        distances.append(pointDistance(point, center))
    minDist = min(distances)
    minDistIndex = distances.index(minDist)
    centralPoint = points[minDistIndex]

    # Calculate distance from center point to all other points
    radii = []
    for point in points:
        radii.append(pointDistance(point, centralPoint))
    
    # Return radii results
    return radii

# Display ring image from calculated radii
def ShowRingFromRadii(radii):
    fig, ax = plt.subplots()
    for radius in radii:
        circle = ptchs.Circle( (0,0), radius, fill=False)
        ax.add_artist(circle)
    ax.set_xlim(-1 * max(radii) - 100, max(radii) + 100)
    ax.set_ylim(-1 * max(radii) - 100, max(radii) + 100)
    ax.set_aspect(1)
    plt.show()

# Full workflow from image to ring plot
def SpotsToRing(image):
    image = CenterImage(image)
    radii = CalculateRadii(image)
    ShowRingFromRadii(radii)

def main():
    simImage = skimage.io.imread("PuO_01-3_SinglePoint.png", as_gray=True)
    expImage = skimage.io.imread("Exp_PuO_01-3_SinglePoint.png", as_gray=True)

    SpotsToRing(simImage)
    SpotsToRing(expImage)

if __name__ == "__main__":
    main()