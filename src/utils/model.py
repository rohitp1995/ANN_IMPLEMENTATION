import tensorflow as tf
import time
import os


def get_model (LOSS, OPTIMIZER, METRICS):

    Layers = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300,activation='relu' ,name="FirstLayer"),
          tf.keras.layers.Dense(100,activation='relu' ,name="SecondLayer"),
          tf.keras.layers.Dense(10,activation='softmax' ,name="output")]

    model_clf = tf.keras.models.Sequential(Layers)

    model_clf.compile(loss='sparse_categorical_crossentropy',
                          optimizer = 'SGD',metrics='accuracy')

    return model_clf


def get_unique_filename(filename):
      unique_name = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
      return unique_name


def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)


