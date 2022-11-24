import tensorflow as tf

def CreateRadiiDNN(numClasses):
    model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(numClasses)
    ])

    return model