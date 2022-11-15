from email.mime import base
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from skimage import io
import time
import csv
from DataHelp.ImageVisualization import PreviewDataSet
from models.GenericCNN import CreateGenericCNN

def main():
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

if __name__ == "__main__":
    main()
