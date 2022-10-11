import csv
import os

srcDir = "/Users/mitchellmika/Desktop/CrystalDataUnsorted"
desDir = "/Users/mitchellmika/Desktop/CrystalDataSorted"

with open("CrystalDirStructure.csv", mode="r") as file:
    csvFile = csv.reader(file)

    for line in csvFile:
        os.popen(f'cp "{srcDir}/{line[0]}.jpg" "{desDir}/{line[1]}/"')