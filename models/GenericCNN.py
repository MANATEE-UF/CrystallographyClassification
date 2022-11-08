import tensorflow as tf

def CreateGenericCNN(imageHeight, imageWidth, numClasses):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, 7, padding="same", activation="relu", input_shape=(imageHeight, imageWidth, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(numClasses))

    return model