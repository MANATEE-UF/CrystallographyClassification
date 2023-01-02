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
import LatticeUtils

# TODO: Implement scale in order to change from sim to exp

# Take structured directory containing labeled subdirectories with image and create csv to be read into tf dataset
def GenerateCsv(inDir, outDir, trials=10):
    # Create 2D array that holds class and list of radii
    trainData = []
    uniqueClasses = []
    numSubDirs = len(os.listdir(inDir))
    subDirCount = 1
    for subdir in os.listdir(inDir):
        print(f"Processing sub-directory {subdir}")
        imageCount = 1
        try:
            uniqueClasses.append(subdir)
            if len(os.listdir(f"{inDir}/{subdir}")) < 5:
                print(f"{subdir} of insufficient size for training (< 5), skipping")
                subDirCount += 1
                continue
            for image in os.listdir(f"{inDir}/{subdir}"):
                try:
                    trialParams = []
                    for i in range(trials):
                        print(f"Sub Dir {subDirCount}/{numSubDirs}, Image {imageCount}/{len(os.listdir(f'{inDir}/{subdir}'))}, Trial {i}/{trials}")
                        readImage = skimage.util.img_as_float(skimage.io.imread(f"{inDir}/{subdir}/{image}", as_gray=True))
                        maximaCoords = Utils.GetImageMaxima(readImage)
                        p1, p2, p3 = LatticeUtils.FindLattice(maximaCoords)
                        trialParams.append(list(LatticeUtils.GetLatticeParams(p1, p2, p3)))
                    params = list(np.median(trialParams, 0))
                    params = [subdir] + params
                    trainData.append(params)
                except:
                    print(f"Unable to process {image}")
                imageCount += 1
        except NotADirectoryError:
            pass
        subDirCount += 1

    testData = []
    for label in uniqueClasses:
        indices = np.where(np.array(trainData)[:,0]==label)[0]
        numItemsInClass = len(indices)
        numItemsInTestSet = 1 if numItemsInClass >= 5 else 0
        indicesToPullTest = list(np.random.choice(indices, numItemsInTestSet, replace=False))
        indicesToPullTest.sort(reverse=True)

        for index in indicesToPullTest:
            testData.append(trainData.pop(index))
    
    with open(f"{outDir}/TrainingData.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(trainData)
    
    with open(f"{outDir}/TestingData.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(testData)

def main():
    inDir = "/Users/mitchellmika/Desktop/ExpSortedZonedReduced"
    outDir = "LatticeTraining/Data/ExpZoneReducedThreeFeatures"

    GenerateCsv(inDir, outDir)

if __name__ == "__main__":
    main()