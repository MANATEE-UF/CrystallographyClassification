import numpy as np
import skimage
from scipy import ndimage as ndi
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
from Utils import *
import csv

# Find maxima of TEM image
def ConvertImageToMaxima(image, showPoints=False):
    image = skimage.filters.median(image) # Median filter to despeckle image before finding points
    image -= np.mean(image) * 2
    image *= 2
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    image = skimage.filters.median(image)
    image = skimage.exposure.adjust_gamma(image, 0.6)

    coords = skimage.feature.peak_local_max(image, min_distance=20, threshold_rel=0.5) # TODO: Justify pre-processing steps

    peakMask = np.zeros_like(image, dtype=bool)
    peakMask[tuple(coords.T)] = True

    if showPoints:
        plt.imshow(image, cmap="gray")
        plt.scatter(coords[:,1], coords[:,0])
        plt.show()

    return peakMask * 1

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
def CalculateRadii(image, scale=1):
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
        radii.append(pointDistance(point, centralPoint) * scale) # Scale is 1/nm per pixel
    
    # Return radii results
    return radii

# Display ring image from calculated radii
def ShowRingFromRadii(radii, saveImage=None):
    fig, ax = plt.subplots()
    for radius in radii:
        circle = ptchs.Circle( (0,0), radius, fill=False)
        ax.add_artist(circle)
    ax.set_xlim(-1 * max(radii) - 100, max(radii) + 100)
    ax.set_ylim(-1 * max(radii) - 100, max(radii) + 100)
    ax.set_aspect(1)

    if saveImage is not None:
        plt.savefig(saveImage)
    else:
        plt.show()
    
    plt.clf()
    plt.close()

# Full workflow from image to radii
def SpotsToRadii(image, showSteps=False, scale=1):
    image = ConvertImageToMaxima(image, showSteps)
    image = CenterImage(image, showSteps)
    radii = CalculateRadii(image, scale)

    return radii

def main():
    inDir = "/Users/mitchellmika/Desktop/In"
    outDir = "/Users/mitchellmika/Desktop/Out"
    csvFile = f"{outDir}/radii.csv"

    for image in os.listdir(inDir):
        try:
            imageName = image
            savePath = f"{outDir}/{image}"
            image = f"{inDir}/{image}"
            image = skimage.util.img_as_float(skimage.io.imread(image, as_gray=True))
            scaleFactor = (1.0 / 29.8) * 100 # 298 pixels = 10 1/nm for experimental images. Factor of 100 is applied to space out the rings
            radii = SpotsToRadii(image, showSteps=False, scale=scaleFactor)
            radii.sort()

            ShowRingFromRadii(radii, saveImage=savePath) 

            with open(csvFile, "a") as f:
                radii = [imageName] + radii
                writer = csv.writer(f)
                writer.writerow(radii)
        except ValueError:
            print(f"Cannot open {image}")

if __name__ == "__main__":
    main()