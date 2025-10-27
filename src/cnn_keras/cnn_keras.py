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


        self.model = tf.keras.Sequential([

                Input(shape=(input_size, input_size, input_channels)),

                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.1),
                RandomTranslation(0.1, 0.1),
 
                Conv2D(16, (3, 3), activation='relu'), 
                BatchNormalization(), 
                MaxPooling2D((2, 2)),   

                Conv2D(32, (3, 3), activation='relu'), 
                BatchNormalization(), 
                MaxPooling2D((2, 2)), 

                Conv2D(64, (3, 3), activation='relu'), 
                BatchNormalization(), 
                MaxPooling2D((2, 2)), 

                                
                Flatten(), 
                Dense(18), 
            ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                          metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=2)])


        # -------------------------------------- 
        # MODEL 0

        model_0 = Sequential()
        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, input_channels)))

        model_0.add(MaxPooling2D(pool_size=(2, 2)))

        # model_0.add(Dropout(0.5))

        model_0.add(Flatten())
        model_0.add(Dense(18, activation='sigmoid'))

        # --------------------------------------
        # MODEL 1

        model_1 = Sequential()
        model_1.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_1.add(MaxPooling2D(pool_size=(2, 2)))

        model_1.add(Dropout(0.25))

        model_1.add(Flatten())
        model_1.add(Dense(18, activation='sigmoid'))

        # --------------------------------------
        # MODEL 2

        model_2 = Sequential()
        model_2.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_2.add(MaxPooling2D(pool_size=(2, 2)))

        # model_2.add(Dropout(0.25))

        model_2.add(Flatten())
        model_2.add(Dense(18, activation='sigmoid'))

        # --------------------------------------
        # MODEL 3

        model_3 = Sequential()
        model_3.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_3.add(MaxPooling2D(pool_size=(2, 2)))

        # model_3.add(Dropout(0.25))

        model_3.add(Flatten())
        model_3.add(Dense(18, activation='sigmoid'))

        # --------------------------------------

        # MODEL 4

        model_4 = Sequential()
        model_4.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_4.add(MaxPooling2D(pool_size=(2, 2)))

        model_4.add(Dropout(0.25))

        model_4.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model_4.add(MaxPooling2D(pool_size=(2, 2)))

        model_4.add(Dropout(0.25))

        model_4.add(Flatten())
        model_4.add(Dense(18, activation='sigmoid'))

        # --------------------------------------

        # MODEL 5

        model_5 = Sequential()
        model_5.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_5.add(MaxPooling2D(pool_size=(2, 2)))

        model_5.add(Dropout(0.25))

        model_5.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model_5.add(MaxPooling2D(pool_size=(2, 2)))

        model_5.add(Dropout(0.25))

        model_5.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model_5.add(MaxPooling2D(pool_size=(2, 2)))

        model_5.add(Flatten())
        model_5.add(Dense(18, activation='sigmoid'))

        # --------------------------------------

        # MODEL 6

        model_6 = Sequential()
        model_6.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(input_size, input_size, input_channels)))
        model_6.add(MaxPooling2D(pool_size=(2, 2)))

        model_6.add(Dropout(0.25))

        model_6.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model_6.add(MaxPooling2D(pool_size=(2, 2)))

        model_6.add(Dropout(0.25))

        model_6.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model_6.add(MaxPooling2D(pool_size=(2, 2)))

        model_6.add(Flatten())

        model_6.add(Dense(128, activation='relu'))
        model_6.add(Dropout(0.5))

        model_6.add(Dense(18, activation='sigmoid'))

        # --------------------------------------

        self.models = [model_0, model_1, model_2, model_3, model_4, model_5, model_6, self.model]

        # for model in self.models:
        #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #     model.summary()

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






