# region Python
import pathlib
# endregion

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
# endregion


class CNNService:
    # region Parameters
    __logger: Logger
    # endregion

    def __init__(self, logger: Logger):
        self.__logger = logger
    # end __init__()

    def load_data(self, path: str, batch_size: int = 32, img_height: int = 384, img_width: int = 384):
        # region Loading data folder
        data_dir = pathlib.Path(path)

        image_count = len(list(data_dir.glob('*/*.png')))
        self.__logger.debug(image_count)
        # endregion

        # region Create dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names
        self.__logger.debug(class_names)
        # endregion

        return train_ds, val_ds
    # end load_data()

    def build_cnn_model(self, activation, input_shape=(384, 384, 3)):
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
    # end build_cnn_model()

    # noinspection PyMethodMayBeStatic
    def compile_and_fit_model(self, model, train_ds, val_ds, batch_size: int = 32, n_epochs: int = 10):
        # Configure the dataset for performance
        autotune = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        # compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='healthcare_model.h5',
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True)]

        # fit the model
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_ds)

        return model, history
    # end compile_and_fit_model()
