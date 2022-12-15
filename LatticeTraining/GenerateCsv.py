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

# Take structured directory containing labeled subdirectories with image and create csv to be read into tf dataset
# FIXME: Test csv is empty
def GenerateCsv(inDir, outDir, trials=10):
    # Create 2D array that holds class and list of radii
    trainData = []
    uniqueClasses = []
    for subdir in os.listdir(inDir):
        print(f"Processing sub-directory {subdir}")
        count = 1
        try:
            uniqueClasses.append(subdir)
            for image in os.listdir(f"{inDir}/{subdir}"):
                try:
                    trialParams = []
                    for i in range(trials):
                        print(f"Processing {count}/{len(os.listdir(f'{inDir}/{subdir}'))}, Trial {i}/{trials}")
                        readImage = skimage.util.img_as_float(skimage.io.imread(f"{inDir}/{subdir}/{image}", as_gray=True))
                        maximaCoords = Utils.GetImageMaxima(readImage)
                        p1, p2, p3 = LatticeUtils.FindLattice(maximaCoords)
                        trialParams.append(list(LatticeUtils.GetLatticeParams(p1, p2, p3)))
                    params = list(np.median(trialParams, 0))
                    params = [subdir] + params
                    trainData.append(params)
                except ValueError:
                    print(f"Unable to process {image}")
                count += 1
        except NotADirectoryError:
            pass

    testData = []
    for label in uniqueClasses:
        indices = np.where(np.array(trainData)[:,0]==label)[0]
        numItemsInClass = len(indices)
        numItemsInTestSet = int(numItemsInClass * 0.1) if int(numItemsInClass * 0.1) >= 1 else 0
        indicesToPullTest = np.random.choice(indices, numItemsInTestSet, replace=False)

        for index in indicesToPullTest:
            testData.append(trainData.pop(index))
    
    with open(f"{outDir}/TrainingData.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(trainData)
    
    with open(f"{outDir}/TestingData.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(testData)

def main():
    inDir = "/Users/mitchellmika/Desktop/ExpSorted"
    outDir = "LatticeTraining/Data/RawExpNoZone"

    GenerateCsv(inDir, outDir)

if __name__ == "__main__":
    main()