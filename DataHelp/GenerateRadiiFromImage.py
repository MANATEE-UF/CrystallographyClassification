import numpy as np
import skimage
from scipy import ndimage as ndi
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
import csv

def Translate(image, xTransform, yTransform):
    tr = skimage.transform.EuclideanTransform(translation=[xTransform, yTransform])
    img = skimage.transform.warp(image, tr)
    return img

def GetImageSize(image, path=True):
    # Get image dimensions before processing further
    if path:
        image = skimage.io.imread(image)
    imageHeight = len(image)
    imageWidth = len(image[0])

    return imageHeight, imageWidth

# Find maxima of TEM image
def ConvertImageToMaxima(image, showScatter=False, saveScatter=False, imgName="default"):
    image = skimage.filters.median(image) # Median filter to despeckle image before finding points
    image -= np.mean(image) * 2
    image *= 2
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    image = skimage.filters.median(image)
    image = skimage.exposure.adjust_gamma(image, 0.6)

    coords = skimage.feature.peak_local_max(image, min_distance=40, threshold_rel=0.5, num_peaks=21) # TODO: Justify pre-processing steps and num_peaks

    peakImage = np.zeros_like(image, dtype=float)
    peakImage[tuple(coords.T)] = 1.0

    if showScatter:
        plt.imshow(image, cmap="gray")
        plt.scatter(coords[:,1], coords[:,0])
        plt.show()
    elif saveScatter:
        plt.imshow(image, cmap="gray")
        plt.scatter(coords[:,1], coords[:,0])
        plt.savefig(f"{imgName}_scatter.jpg")
        plt.clf()
        plt.close()

    return peakImage

# Centers an image based on position of dots
# Not used in standard workflow because translation blurs the image
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

# Fill array to certain size with existing values in order to keep consistent input size
def FillArrayToSize(arr, size):

    if len(arr) < size:
        numRepetitions = int(size / len(arr)) + 1
        arr *= numRepetitions
        arr = arr[0:size]
    
    return arr

# Calculate radius from central most point
def CalculateRadii(image, scale=1):
    # Find all points in image
    rows, cols = np.where(image != 0)
    points = list(zip(cols,rows))

    # Find centroid of points
    xAvg = np.average(cols)
    yAvg = np.average(rows)
    centroid = (xAvg, yAvg)

    # Find point closest to centroid
    pointDistance = lambda p1,p2: math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) # Accepts (x,y) tuple
    distanceToCentroid = []
    for point in points:
        distanceToCentroid.append(pointDistance(point,centroid))
    minDist = min(distanceToCentroid)
    minDistIndex = distanceToCentroid.index(minDist)
    centralPoint = points[minDistIndex]

    # Calculate distance from center point to all other points
    radii = []
    for point in points:
        radii.append(pointDistance(point, centralPoint) * scale) # Scale is 1/nm per pixel
    
    # Process radii results
    radii.sort()
    radii.pop(0) # Removes 0 radius from list
    radii = FillArrayToSize(radii, 20)

    return radii

# Display ring image from calculated radii
def ShowRingFromRadii(radii, saveImage=False, imgName="default"):
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

    if saveImage:
        plt.savefig(f"{imgName}_ring.jpg")
    else:
        plt.show()
    
    plt.clf()
    plt.close()

# Full workflow from image to radii
def ImageToRadii(image, scale=1, showSteps=False,  saveRing=False, saveScatter=False, imgName="default"):
    image = ConvertImageToMaxima(
        image, 
        showScatter=showSteps, 
        saveScatter=saveScatter, 
        imgName=imgName
    )  
    radii = CalculateRadii(image, scale)
    
    if showSteps or saveRing:
        ShowRingFromRadii(radii, saveImage=saveRing, imgName=imgName)

    return radii

# Pre-process directory of TEM images
# Saves radii values to csv file and optionally saves a ring image and scatter plot of found maxima for debugging
def PreProcessDir(inDir, outDir, showSteps=False, saveRing=False, saveScatter=False):
    csvFile = f"{outDir}/radii.csv"

    for imageName in os.listdir(inDir):
        try:
            image = skimage.util.img_as_float(skimage.io.imread(f"{inDir}/{imageName}", as_gray=True))
            scaleFactor = 1.0 / 29.8 # 298 pixels = 10 1/nm for experimental images
            
            radii = ImageToRadii(image,
                scale=scaleFactor, 
                showSteps=showSteps,  
                saveRing=saveRing, 
                saveScatter=saveScatter, 
                imgName=f"{outDir}/{os.path.splitext(imageName)[0]}"
            )
            
            radii.sort()

            with open(csvFile, "a") as f:
                radii = [imageName] + radii
                writer = csv.writer(f)
                writer.writerow(radii)
        
        except ValueError:
            print(f"Cannot open {image}")

# Take structured directory containing labeled subdirectories with image and create csv to be read into tf dataset
def GenerateCsvFromStructuredDir(inDir, csvPath):
    features = []
    for subdir in os.listdir(inDir):
        print(f"Processing sub-directory {subdir}")
        count = 1
        try:
            for image in os.listdir(f"{inDir}/{subdir}"):
                try:
                    print(f"Processing {count}/{len(os.listdir(f'{inDir}/{subdir}'))}")
                    readImage = io.imread(f"{inDir}/{subdir}/{image}", as_gray=True)
                    scaleFactor = 1.0 / 29.8
                    radii = ImageToRadii(readImage, scaleFactor)
                    radii = [subdir] + radii
                    features.append(radii)
                except ValueError:
                    print(f"Unable to process {image}")
                count += 1
        except NotADirectoryError:
            pass

    with open(csvPath, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)
    
def main():
    inDir = "/Users/mitchellmika/Desktop/In"
    GenerateCsvFromStructuredDir(inDir, "./LabelledRadii.csv")

if __name__ == "__main__":
    main()