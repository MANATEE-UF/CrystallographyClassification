import tensorflow as tf

def CreateRadiiDNN(numClasses):
    model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(40),
        tf.keras.layers.Dense(40),
        tf.keras.layers.Dense(numClasses, activation="softmax")
    ])

    return model