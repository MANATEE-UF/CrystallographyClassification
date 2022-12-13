import sys
import os
if os.path.basename(os.getcwd()).upper() == "CRYSTALLOGRAPHYCLASSIFICATION":
    sys.path.append(".")
    sys.path.append("./RadiiTraining")
elif os.path.basename(os.getcwd()).upper() == "RADIITRAINING":
    sys.path.append("..")
import skimage
import Utils
import RadiiUtils

# Saves a ring image and/or maxima scatter plot of found images in inDir to outDir for debugging
def RingAndMaximaPlotter(inDir, outDir, saveRing=False, saveScatter=False):
    for imageName in os.listdir(inDir):
        try:
            image = skimage.util.img_as_float(skimage.io.imread(f"{inDir}/{imageName}", as_gray=True))
            scaleFactor = 1.0 / 29.8 # 298 pixels = 10 1/nm for experimental images

            maximaCoords = Utils.GetImageMaxima(image, maxPeaks=100)
            if saveScatter:
                Utils.SaveMaximaPlot(maximaCoords, image, saveName=f"{outDir}/{os.path.splitext(imageName)[0]}_scatter.jpg")

            if saveRing:
                radii = RadiiUtils.CalculateRadii(maximaCoords, scale=scaleFactor, fill=None)
                RadiiUtils.SaveRadiiPlot(radii, saveName=f"{outDir}/{os.path.splitext(imageName)[0]}_ring.jpg")

        except ValueError:
            print(f"Cannot open {imageName}")

def main():
    inDir = "/Users/mitchellmika/Desktop/FourZones"
    outDir = "/Users/mitchellmika/Desktop/Out"

    RingAndMaximaPlotter(inDir, outDir, saveRing=False, saveScatter=True)

if __name__ == "__main__":
    main()