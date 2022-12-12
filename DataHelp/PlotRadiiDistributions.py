import matplotlib.pyplot as plt
import numpy as np
import csv

def Scatter1D(csvPath,saveFig=""):
    colors = ["black", "lightcoral", "darkred", "peru", "orange", "green", "cyan", "navy", "violet", "deeppink"]

    fig, ax = plt.subplots()
    with open(csvPath, mode="r") as file:
        csvFile = csv.reader(file)

        labels = []
        for line in csvFile:
            if line[0] not in labels:
                labels.append(line[0])
            y = labels.index(line[0])
            ax.scatter(list(map(float, line[1:])), np.ones(len(line[1:])) * y, c=colors[y], s=0.1)
    
    ax.set_xlabel("Radius")
    ax.set_xlim(0, 30)
    ax.set_ylabel("Class")
    ax.set_yticks(range(0, len(labels)), labels)
    ax.set_ylim(-0.5, len(labels)-0.5)
    
    plt.tight_layout()
    if saveFig:
        plt.savefig(saveFig)
    else:
        plt.show()

def main():
    Scatter1D("FourZones_NoFillNoMax/TrainingData.csv")

if __name__ == "__main__":
    main()