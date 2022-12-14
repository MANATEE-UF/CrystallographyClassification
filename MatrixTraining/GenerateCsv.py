import sys
if os.path.basename(os.getcwd()).upper() == "CRYSTALLOGRAPHYCLASSIFICATION":
    sys.path.append(".")
    sys.path.append("./abTraining")
elif os.path.basename(os.getcwd()).upper() == "MATRIXTRAINING":
    sys.path.append("..")
import numpy as np
import skimage
from scipy import ndimage as ndi
import os
import math
import matplotlib.pyplot as plt
import csv
import Utils