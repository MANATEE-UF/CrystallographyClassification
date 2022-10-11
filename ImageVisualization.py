import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#########################################################################################################
# PreviewDataSet()
#
# Displays each image in a dataset
#
# Parameters:
#   dataSet:        (tf.tensor) tf.tensors with shape (numImages, imageHeight, imageWidth, channels)
#   viewTime:       (float)     Value indicating how long each image will be displayed
#
# Returns:
#   None
#########################################################################################################
def PreviewDataSet(dataSet, viewTime):
    for i in range(len(dataSet)):
        img = dataSet[i]
        img = img.numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"Series {i}/{len(dataSet)}")
        plt.show(block=False)
        plt.pause(viewTime)
        plt.close("all")