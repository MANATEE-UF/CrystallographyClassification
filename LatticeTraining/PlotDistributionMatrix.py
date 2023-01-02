# TODO: Make plots of different 3D spaces (e.g. (a,b,theta), (a,b,d), (a,theta,d), etc.)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import csv

def ShowLatticeScatter(csvPath):
    fig = plt.figure(figsize=(10,7))
    axs = plt.axes(projection="3d")
    colors = ["black", "lightcoral", "saddlebrown", "orange", "lawngreen", "cyan", "indigo", "magenta", "navy"]

    with open(csvPath, mode="r") as file:
        csvFile = csv.reader(file)

        labels = []
        for line in csvFile:
            if line[0] not in labels:
                labels.append(line[0])
            axs.scatter3D(float(line[1]), float(line[2]), float(line[4]), color=colors[labels.index(line[0])])
    
    plt.show()

def main():
    ShowLatticeScatter("LatticeTraining/Data/FourZones/TrainingData.csv")

if __name__ == "__main__":
    main()