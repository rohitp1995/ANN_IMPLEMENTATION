import tensorflow as tf


def get_model (LOSS, OPTIMIZER, METRICS):

    Layers = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300,activation='relu' ,name="FirstLayer"),
          tf.keras.layers.Dense(100,activation='relu' ,name="SecondLayer"),
          tf.keras.layers.Dense(10,activation='softmax' ,name="output")]

    model_clf = tf.keras.models.Sequential(Layers)

    model_clf.compile(loss='sparse_categorical_crossentropy',
                          optimizer = 'SGD',metrics='accuracy')

    return model_clf
