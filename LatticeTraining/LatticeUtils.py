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
    # TODO: Write a better test than slope test, causes undefined behavior for vertical lines (maybe use band of lines)
    # TODO: Prove that picking the second closest point is valid way of finding the lattice
    secondPoint = None
    minDist = 1E9
    for point in points:
        if abs(Utils.SlopeEq2D(point, centralPoint) - Utils.SlopeEq2D(firstPoint, centralPoint)) < 0.3:
            continue
        dist = Utils.DistanceEq2D(point, centralPoint)

        if dist < minDist:
            minDist = dist
            secondPoint = point
    
    return list(centralPoint), list(firstPoint), list(secondPoint) # Convert to list so that its mutable in later steps

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

    # Find angle between vectors, normalize to [0,2pi], find smallest angle in the parallelogram
    # FIXME: Hexagonal lattice just as likely to give 120 as it is 60
    dotProduct = firstPoint[0] * secondPoint[0] + firstPoint[1] * secondPoint[1]
    theta = math.degrees(math.acos(dotProduct / (dMin * dMax)))
    theta = theta/2 if theta > 100 else theta

    # Find absolute value of lattice determinant (volume)
    determinant = abs( ( firstPoint[0] * secondPoint[1] ) - ( firstPoint[1] * secondPoint[0] ) )

    return dMin, dMax, theta, determinant

# Test to show if lattices are being found correctly
def TestLatticeFinder(trials=3):
    fig,axs = plt.subplots(3, trials)
    for i in range(1,4):
        readImage = skimage.util.img_as_float(skimage.io.imread(f"LatticeTest{i}.jpg", as_gray=True))
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
        readImage = skimage.util.img_as_float(skimage.io.imread(f"LatticeTest{i}.jpg", as_gray=True))
        print(f"Begin LatticeTest{i}.jpg")
        for j in range(trials):
            maxima = Utils.GetImageMaxima(readImage)
            p1, p2, p3 = FindLattice(maxima)
            dMin, dMax, theta, determinant = GetLatticeParams(p1, p2, p3)
            print(f"{dMin:.2f}, {dMax:.2f}, {theta:.2f}, {determinant:.2f}")
        print("-" * 15)

# FIXME: Lattice choices can vary for a given lattice using current algorithm
def main():
    #TestLatticeFinder(3)
    TestLatticeParams(10)

if __name__ == "__main__":
    main()