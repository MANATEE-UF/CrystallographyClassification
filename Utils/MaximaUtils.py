import skimage
import numpy as np
import matplotlib.pyplot as plt

# Find maxima of TEM image
def GetImageMaxima(image, maxPeaks=float('inf')):
    image = skimage.filters.median(image)
    image -= np.mean(image) * 2
    image *= 2
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    image = skimage.filters.median(image)
    image = skimage.exposure.adjust_gamma(image, 0.6)

    coords = skimage.feature.peak_local_max(image, min_distance=40, threshold_rel=0.5, num_peaks=maxPeaks) # TODO: Justify pre-processing steps and num_peaks

    return coords

# Display maxima plot to screen
def ShowMaximaPlot(maximaCoords, image):
    plt.imshow(image, cmap="gray")
    plt.scatter(maximaCoords[:,1], maximaCoords[:,0])
    plt.show()

# Save maxima plot to saveName path
def SaveMaximaPlot(maximaCoords, image, saveName):
    plt.imshow(image, cmap="gray")
    plt.scatter(maximaCoords[:,1], maximaCoords[:,0])
    plt.savefig(f"{saveName}")
    plt.clf()
    plt.close()
