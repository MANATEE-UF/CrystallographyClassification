import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from skimage import io
import time
import pandas as pd
import csv
from models.GenericCNN import CreateGenericCNN
from models.DNN import CreateDNN

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# FIXME: Auto generate test set rather than using a DataToPredict directory
def RunCNN(saveModel=False):
    imageHeight = 200 # This is going to distort the image, which will make classifying based on point distance very difficult (impossible)
    imageWidth = 200

    # Save images for prediction
    images = {}
    for image in os.listdir("./DataToPredict/"):
        img = tf.keras.utils.load_img(
            f"./DataToPredict/{image}",
            target_size=(imageHeight, imageWidth),
            color_mode="grayscale"
        )

        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, 0)

        images[image] = img

    # Get image data from directory and store in tf.dataset
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory="/Users/mitchellmika/Desktop/CrystalDataSortedZoned",
        validation_split=0.2,
        subset="both",
        color_mode="grayscale",
        image_size=(imageHeight, imageWidth),
        seed=123,
        batch_size=64
    )

    class_names = train_ds.class_names
    print(class_names)
    numClasses = len(class_names)

    model = CreateGenericCNN(imageHeight, imageWidth, numClasses)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", min_delta=0.001, patience=3)]
    )
    
    for key in images:
        prediction = model.predict(images[key])

        score = tf.nn.softmax(prediction[0])
        print(f"Image {key} is predicted to be {class_names[np.argmax(score)]} with {100*np.max(score):.2f}% confidence.")

def RunRadiiDNN(saveModel=False):
    trainingCsv = "RadiiTraining/Data/FourZones/TrainingData.csv"
    testCsv = "RadiiTraining/Data/FourZones/TestingData.csv"

    headers = ["Label"]
    for i in range(50):
        headers.append(f"Radius_{i}")

    trainingSet = pd.read_csv(trainingCsv, names=headers)
    testSet = pd.read_csv(testCsv, names=headers)

    trainingLabels = trainingSet.pop("Label")
    testLabels = testSet.pop("Label")
    trainingLabels = np.array(trainingLabels, dtype=str)
    testLabels = np.array(testLabels)

    uniqueLabels = np.unique(trainingLabels)
    numClasses = len(uniqueLabels)

    print()
    print("Trained on the following data:")
    for label in uniqueLabels:
        print(f"{label}: {len(np.where(trainingLabels==label)[0])} instances")

    yTrain = []
    for label in trainingLabels:
        temp = np.zeros(numClasses)
        temp[np.where(uniqueLabels==label)[0][0]] = 1.0
        yTrain.append(temp)
    yTrain = np.array(yTrain)

    yTest = []
    for label in testLabels:
        temp = np.zeros(numClasses)
        temp[np.where(uniqueLabels==label)[0][0]] = 1.0
        yTest.append(temp)
    yTest = np.array(yTest)

    trainingFeatures = trainingSet.copy()
    testFeatures = testSet.copy()

    xTrain = np.array(trainingFeatures)
    
    xTest = np.array(testFeatures)

    model = CreateDNN(numClasses)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # False if softmax used in last layer
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"])
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=200)
    
    def PrintProbs(input):
        string = "[ "
        for i in range(len(input)):
            string += f"{input[i]:.2f}, "
        return string + "]"

    preds = model.predict(xTest)
    print(uniqueLabels)
    for i in range(len(preds)):
        predClass = uniqueLabels[np.argmax(preds[i])]
        trueClass = uniqueLabels[np.argmax(yTest[i])]
        if trueClass == predClass:
            printColor = bcolors.OKGREEN
        else:
            printColor = bcolors.FAIL
        print(f"True: {trueClass:10} {printColor} Predicted: {predClass:10} {100 * np.max(preds[i]):.2f}% {bcolors.ENDC}")#- {PrintProbs(preds[i])} {bcolors.ENDC}")

# TODO: Look into sklearn.preprocessing as method of one hot encoding, pg 67 of ml book
def RunLatticeDNN(saveModel=False):
    trainingCsv = "LatticeTraining/Data/ExpZoneReducedThreeFeatures/TrainingData.csv"
    testCsv = "LatticeTraining/Data/ExpZoneReducedThreeFeatures/TestingData.csv"

    headers = ["Label", "d1", "d2", "theta"]#, "discriminant"]

    trainingSet = pd.read_csv(trainingCsv, names=headers)
    testSet = pd.read_csv(testCsv, names=headers)

    trainingLabels = trainingSet.pop("Label")
    testLabels = testSet.pop("Label")
    trainingLabels = np.array(trainingLabels, dtype=str)
    testLabels = np.array(testLabels)

    uniqueLabels = np.unique(trainingLabels)
    numClasses = len(uniqueLabels)

    print()
    print("Trained on the following data:")
    for label in uniqueLabels:
        print(f"{label}: {len(np.where(trainingLabels==label)[0])} instances")

    # One hot encoding

    # Categorical cross entropy
    # yTrain = []
    # for label in trainingLabels:
    #     temp = np.zeros(numClasses)
    #     temp[np.where(uniqueLabels==label)[0][0]] = 1.0
    #     yTrain.append(temp)
    # yTrain = np.array(yTrain)

    # yTest = []
    # for label in testLabels:
    #     temp = np.zeros(numClasses)
    #     temp[np.where(uniqueLabels==label)[0][0]] = 1.0
    #     yTest.append(temp)
    # yTest = np.array(yTest)

    # Sparse categorical cross entropy
    yTrain = []
    for label in trainingLabels:
        temp = np.where(uniqueLabels==label)[0][0]
        yTrain.append(temp)
    yTrain = np.array(yTrain)

    yTest = []
    for label in testLabels:
        temp = np.where(uniqueLabels==label)[0][0]
        yTest.append(temp)
    yTest = np.array(yTest)


    trainingFeatures = trainingSet.copy()
    testFeatures = testSet.copy()

    xTrain = np.array(trainingFeatures)
    
    xTest = np.array(testFeatures)

    model = CreateDNN(numClasses)
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # False if softmax used in last layer
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"])
    
    model.fit(xTrain, 
        yTrain, 
        validation_data=(xTest, yTest),  
        epochs=200)
    
    def PrintProbs(input):
        string = "[ "
        for i in range(len(input)):
            string += f"{input[i]:.2f}, "
        return string + "]"

    preds = model.predict(xTest)
    #print(uniqueLabels)
    for i in range(len(preds)):
        predClass = uniqueLabels[np.argmax(preds[i])]
        trueClass = uniqueLabels[yTest[i]]
        if trueClass == predClass:
            printColor = bcolors.OKGREEN
        else:
            printColor = bcolors.FAIL
        print(f"True: {trueClass:10} {printColor} Predicted: {predClass:10} {100 * np.max(preds[i]):.2f}% {bcolors.ENDC}")# - {PrintProbs(preds[i])} {bcolors.ENDC}")

def main():
    #RunRadiiDNN()
    RunLatticeDNN()

if __name__ == "__main__":
    main()
