import sys
import os
if os.path.basename(os.getcwd()).upper() == "CRYSTALLOGRAPHYCLASSIFICATION":
    sys.path.append(".")
    sys.path.append("./abTraining")
elif os.path.basename(os.getcwd()).upper() == "LATTICETRAINING":
    sys.path.append("..")
import numpy as np
import skimage
from scipy import ndimage as ndi
import math
import matplotlib.pyplot as plt
import csv
import Utils

# FIXME: More outliers are being generated than i would like, need to refine algorithm

# Return angle between vectors centered at v0
def AngleBetweenVectors(v0, v1, v2):
    dMin = min(Utils.DistanceEq2D(v0, v1), Utils.DistanceEq2D(v0, v2))
    dMax = max(Utils.DistanceEq2D(v0, v1), Utils.DistanceEq2D(v0, v2))

    dotProduct = (v1[0]-v0[0]) * (v2[0]-v0[0]) + (v1[1]-v0[1]) * (v2[1]-v0[1])
    try:
        theta = math.acos(dotProduct / (dMin * dMax))
    except ValueError:
        print(dotProduct)
        print(dMin)
        print(dMax)
        return 180
    theta = math.degrees(theta)

    return theta

# Finds a lattice based on random central point
# Does not perform any coordinate transformations
def FindLattice(maximaCoords):
    # Zip points into ((x1,y1), (x2,y2), ...) array
    rows = maximaCoords[:,0]
    cols = maximaCoords[:,1]
    points = list(zip(cols,rows))

    # Select a central point at random
    # TODO: Eventually need refactoring to ensure drawn without replacement
    index = np.random.choice(len(points), 1)[0]
    centralPoint = points[index]
    points.remove(centralPoint)

    # Find point closest to central point
    firstPoint = None
    minDist = 1E9
    for point in points:
        dist = Utils.DistanceEq2D(point, centralPoint)

        if dist < minDist:
            minDist = dist
            firstPoint = point
        else:
            pass
    points.remove(firstPoint)
    

    # Pick the second closest point that passes lattice checks
    # TODO: Prove that picking the second closest point is valid way of finding the lattice
    secondPoint = None
    minDist = 1E9
    for point in points:
        if AngleBetweenVectors(centralPoint, firstPoint, point) > 150:
            continue
        dist = Utils.DistanceEq2D(point, centralPoint)

        if dist < minDist:
            minDist = dist
            secondPoint = point
    
    return list(centralPoint), list(firstPoint), list(secondPoint) # Convert to list so that its mutable in later steps, probably should think of better way

# Obtain lattice params that will be used for training
def GetLatticeParams(centralPoint, firstPoint, secondPoint):
    # Translate first point
    firstPoint[0] -= centralPoint[0]
    firstPoint[1] -= centralPoint[1]

    # Translate second point
    secondPoint[0] -= centralPoint[0]
    secondPoint[1] -= centralPoint[1]

    # Translate central point
    centralPoint[0] = 0
    centralPoint[1] = 0

    # Find a and b distances
    dMin = min(Utils.DistanceEq2D(centralPoint, firstPoint), Utils.DistanceEq2D(centralPoint, secondPoint))
    dMax = max(Utils.DistanceEq2D(centralPoint, firstPoint), Utils.DistanceEq2D(centralPoint, secondPoint))

    # Find angle between vectors, normalize to [0,2pi]
    theta = AngleBetweenVectors(centralPoint, firstPoint, secondPoint)
    theta = theta/2 if theta > 100 else theta # FIXME: This is really hacky for handling hexagons

    return dMin, dMax, theta

# Test to show if lattices are being found correctly
def TestLatticeFinder(trials=3):
    fig,axs = plt.subplots(3, trials)
    for i in range(1,4):
        readImage = skimage.util.img_as_float(skimage.io.imread(f"LatticeTraining/LatticeTest{i}.jpg", as_gray=True))
        for j in range(trials):
            maxima = Utils.GetImageMaxima(readImage)
            p1, p2, p3 = FindLattice(maxima)

            axs[i-1][j].imshow(readImage, cmap="gray")
            axs[i-1][j].scatter(p1[0], p1[1], c="y",s=10, marker="*")
            axs[i-1][j].scatter(p2[0], p2[1], c="r",s=10)
            axs[i-1][j].scatter(p3[0], p3[1], c="r",s=10)
            axs[i-1][j].set_axis_off()

    plt.tight_layout()
    plt.show()

# Test to see if lattice param calculations are okay
def TestLatticeParams(trials=5):
    for i in range(1,4):
        readImage = skimage.util.img_as_float(skimage.io.imread(f"LatticeTraining/LatticeTest{i}.jpg", as_gray=True))
        print(f"Begin LatticeTest{i}.jpg")
        for j in range(trials):
            maxima = Utils.GetImageMaxima(readImage)
            p1, p2, p3 = FindLattice(maxima)
            dMin, dMax, theta = GetLatticeParams(p1, p2, p3)
            print(f"{dMin:.2f}, {dMax:.2f}, {theta:.2f}")
        print("-" * 15)

# FIXME: Lattice choices can vary for a given lattice using current algorithm
def main():
    TestLatticeFinder(3)
    #TestLatticeParams(10)

if __name__ == "__main__":
    main()