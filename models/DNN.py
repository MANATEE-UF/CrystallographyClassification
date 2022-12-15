import tensorflow as tf

def CreateDNN(numClasses):
    model = tf.keras.Sequential(
    [
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(numClasses, activation="softmax")
    ])

    return model