from GenerateRadiiFromImage import GenerateCsvFromStructuredDir
from PlotRadiiDistributions import Scatter1D
import numpy as np

def main():
    fillRange = [None,10,50,100]
    maxPeakRange = [10,50,100,float('inf')]
    for fill in fillRange:
        for maxPeak in maxPeakRange:
            inDir = "/Users/mitchellmika/Desktop/SixZonesReflections"
            outDir = "tempDir"
            GenerateCsvFromStructuredDir(inDir, outDir, fill, maxPeak)
            Scatter1D("tempDir/TrainingData.csv", f"/Users/mitchellmika/Desktop/RadiiResults/ZoneReflectionsStudy/{fill}Fill{maxPeak}Max.png")

if __name__ == "__main__":
    main()