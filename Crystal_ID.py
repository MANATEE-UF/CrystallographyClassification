from email.errors import InvalidMultipartContentTransferEncodingDefect
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import csv
from ImageVisualization import PreviewDataSet

def CreateModel(imageHeight, imageWidth, numClasses):

    data_aug_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(input_shape=(imageHeight, imageWidth, 1)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ]
    )

    model = tf.keras.Sequential()

    model.add(data_aug_layer)
    model.add(tf.keras.layers.Rescaling(1./255))

    model.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(numClasses))

    return model

def main():
    imageHeight = 224
    imageWidth = 224

    # Save images for prediction
    images = {}
    for image in os.listdir("./DataToPredict/"):
        img = tf.keras.utils.load_img(
            f"./DataToPredict/{image}",
            target_size=(imageHeight, imageWidth),
            color_mode="rgb"
        )

        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, 0)

        images[image] = img

    # Get image data from directory and store in tf.dataset
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory="/Users/mitchellmika/Desktop/CrystalDataSortedTrimmed",
        validation_split=0.2,
        subset="both",
        color_mode="rgb",
        image_size=(imageHeight, imageWidth),
        seed=123,
        batch_size=10
    )

    class_names = train_ds.class_names
    print(class_names)
    numClasses = len(class_names)

    #model = CreateModel(imageHeight, imageWidth, numClasses)
    model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

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
        epochs=1
    )
    
    for key in images:
        prediction = model.predict(images[key])
        score = tf.nn.softmax(prediction[0])

        print(f"Image {key} is predicted to be {class_names[np.argmax(score)]} with {100*np.max(score):.2f}% confidence.")
        

if __name__ == "__main__":
    main()
