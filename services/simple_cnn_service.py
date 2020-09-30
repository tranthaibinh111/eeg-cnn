# region Package (third-party)
# TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
# endregion

# region motor impairment neural disorders
# utils
from utlis import Logger
# services
from .cnn_service import CNNService
# endregion


class SimpleCNNService(CNNService):
    # region Parameters
    __logger: Logger

    # endregion

    def __init__(self, logger: Logger):
        self.__logger = logger
        super().__init__(self.__logger)
    # end __init__()

    # noinspection PyMethodMayBeStatic
    def build_model(self, activation, input_shape=(384, 384, 3)):
        model = Sequential()

        # Standardize the data
        model.add(Rescaling(1. / 255, input_shape=input_shape))

        # 3 Convolution layer with Max polling
        model.add(Conv2D(16, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        # 3 Full connected layer
        model.add(Dense(128, activation=activation, kernel_initializer="he_normal"))
        model.add(Dense(36, activation=activation, kernel_initializer="he_normal"))
        model.add(Dense(4, activation='softmax'))  # 4 classes

        # summarize the model
        model.summary()

        return model
    # end build_model()

    # noinspection PyMethodMayBeStatic
    def compile_and_fit_model(self, model, train_ds, batch_size: int = 32, n_epochs: int = 10):
        # compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='healthcare_model.h5',
                monitor='val_accuracy',
                save_best_only=True)]

        # fit the model
        model.fit(
            train_ds,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks)

        return model
    # end compile_and_fit_model()
