import sys
import os
if os.path.basename(os.getcwd()).upper() == "CRYSTALLOGRAPHYCLASSIFICATION":
    sys.path.append(".")
    sys.path.append("./RadiiTraining")
elif os.path.basename(os.getcwd()).upper() == "RADIITRAINING":
    sys.path.append("..")
import numpy as np
import skimage
import csv
import Utils
import RadiiUtils

# Take structured directory containing labeled subdirectories with image and create csv to be read into tf dataset
# FIXME: Test csv is empty
def GenerateCsv(inDir, outDir, fill=None, maxPeaks=float('inf')):
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
                    print(f"Processing {count}/{len(os.listdir(f'{inDir}/{subdir}'))}")
                    readImage = skimage.util.img_as_float(skimage.io.imread(f"{inDir}/{subdir}/{image}", as_gray=True))
                    scaleFactor = 1.0 / 29.8 # 10 1/nm = 298 pixels for exp images
                    maximaCoords = Utils.GetImageMaxima(readImage, maxPeaks=maxPeaks)
                    radii = RadiiUtils.CalculateRadii(maximaCoords, scaleFactor, fill=fill)
                    radii = [subdir] + radii
                    trainData.append(radii)
                except ValueError:
                    print(f"Unable to process {image}")
                count += 1
        except NotADirectoryError:
            pass

    testData = []
    for label in uniqueClasses:
        indices = np.where(trainData[:][0]==label)[0]
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
    inDir = "/Users/mitchellmika/Desktop/SixZonesReflections"
    outDir = "SixZonesReflections"

    GenerateCsv(inDir, outDir,fill=50)

if __name__ == "__main__":
    main()