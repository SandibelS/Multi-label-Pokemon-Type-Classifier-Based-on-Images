import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation,
    Input
)

import tensorflow as tf

# import gc
# gc.collect()
# tf.keras.backend.clear_session()


class CNN_keras():

    def __init__(self, input_size : int, input_channels : int):

        # -------------------------------------- 
        # MODEL 0

        model_0 = Sequential()
        model_0.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, input_channels)))

        model_0.add(MaxPooling2D(pool_size=(2, 2)))

        model_0.add(Flatten())
        model_0.add(Dense(18, activation='sigmoid'))

        # --------------------------------------
        # MODEL 1

        model_1 = Sequential()
        model_1.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_1.add(MaxPooling2D(pool_size=(2, 2)))

        model_1.add(Flatten())
        model_1.add(Dense(18, activation='sigmoid'))

        # --------------------------------------
        # MODEL 2

        model_2 = Sequential()
        model_2.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_2.add(MaxPooling2D(pool_size=(2, 2)))

        model_2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model_2.add(MaxPooling2D(pool_size=(2, 2)))

        model_2.add(Flatten())
        model_2.add(Dense(18, activation='sigmoid'))

        # --------------------------------------

    
        self.models = [model_0, model_1, model_2]

        for model in self.models:
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                          metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=2)])


    def train(self, model_id, X_train, y_train, X_test, y_test, batch_size=32, epochs=10):

        model = self.models[model_id]
        model.summary() 
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size)

        return history

    def predict(self, model_id, img, img_size, classes):

        model = self.models[model_id]

        proba = model.predict(img.reshape(1, img_size, img_size, 3))
        top_2 = np.argsort(proba[0])[:-3:-1]  

        for i in range(2):
            print("{}".format(classes[top_2[i]]) + " ({:.3})".format(proba[0][top_2[i]]))
    
    def evaluate(self, model_id, X_test, y_test):

        model = self.models[model_id]
        loss, accuracy = model.evaluate(X_test, y_test)

        return loss, accuracy






