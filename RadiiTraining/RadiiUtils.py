import sys
import os
if os.path.basename(os.getcwd()).upper() == "CRYSTALLOGRAPHYCLASSIFICATION":
    sys.path.append(".")
    sys.path.append("./RadiiTraining")
elif os.path.basename(os.getcwd()).upper() == "RADIITRAINING":
    sys.path.append("..")
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import Utils

# Calculate radius from central most point
def CalculateRadii(maximaCoords, scale=1, fill=None):
    # Find all points in image
    rows = maximaCoords[:,0]
    cols = maximaCoords[:,1]
    points = list(zip(cols,rows))

    # Find centroid of points
    xAvg = np.average(cols)
    yAvg = np.average(rows)
    centroid = (xAvg, yAvg)

    # Find point closest to centroid
    distanceToCentroid = []
    for point in points:
        distanceToCentroid.append(Utils.DistanceEq2D(point,centroid))
    minDist = min(distanceToCentroid)
    minDistIndex = distanceToCentroid.index(minDist)
    centralPoint = points[minDistIndex]

    # Calculate distance from center point to all other points
    radii = []
    for point in points:
        radii.append(Utils.DistanceEq2D(point, centralPoint) * scale) # Scale is 1/nm per pixel
    
    # Process radii results
    radii.sort()
    radii.pop(0) # Removes 0 radius from list
    if fill is not None:
        radii = Utils.FillArrayToSize(radii, fill)

    return radii

# Display ring image from calculated radii
def ShowRadiiPlot(radii):
    fig, ax = plt.subplots()
    for radius in radii:
        circle = ptchs.Circle( (0,0), radius, fill=False)
        ax.add_artist(circle)
    ax.set_xlabel("1/nm")
    ax.set_xlim(-40, 40) # Should capture the largest possible radii based on experimental image sizes
    ax.set_ylabel("1/nm")
    ax.set_ylim(-40, 40)
    ax.set_aspect(1)
    ax.grid(True, alpha=0.5)

    plt.show()

# Save ring image from calculated radii to saveName path
def SaveRadiiPlot(radii, saveName):
    fig, ax = plt.subplots()
    for radius in radii:
        circle = ptchs.Circle( (0,0), radius, fill=False)
        ax.add_artist(circle)
    ax.set_xlabel("1/nm")
    ax.set_xlim(-40, 40) # Should capture the largest possible radii based on experimental image sizes
    ax.set_ylabel("1/nm")
    ax.set_ylim(-40, 40)
    ax.set_aspect(1)
    ax.grid(True, alpha=0.5)

    plt.savefig(f"{saveName}")   
    plt.clf()
    plt.close()