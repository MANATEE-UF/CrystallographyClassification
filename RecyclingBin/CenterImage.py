import Utils
import numpy as np
import matplotlib.pyplot as plt
import math

# Centers an image based on position of dots
# Not used in standard workflow because translation blurs the image
def CenterImage(image, showTranslate=False):

    # Find center point
    height, width = Utils.GetImageSize(image, path=False)
    center = (width/2, height/2)

    # Find all points in image
    rows, cols = np.where(image != 0)

    # Find centroid and calculate appropriate transformation
    xAvg = np.average(cols)
    yAvg = np.average(rows)

    xTrans = xAvg - center[0]
    yTrans = yAvg - center[1]

    # Translate image to center
    image = Utils.TranslateImage(image, xTrans, yTrans)

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

    image = Utils.TranslateImage(image, xTrans, yTrans)

    # Display new image with central point (red) and center of image (blue)
    if showTranslate:
        plt.imshow(image, cmap="gray")
        plt.scatter(centralPoint[0], centralPoint[1], s=50, c="r")
        plt.scatter(center[0], center[1], s=50, c="b")
        plt.show()

    return image
