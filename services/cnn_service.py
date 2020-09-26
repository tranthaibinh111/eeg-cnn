# region Python
import pathlib
# endregion

# region Package (third-party)
# TensorFlow and tf.keras
import tensorflow as tf
# endregion

# region Matplotlib
import matplotlib.pyplot as plt
# endregion
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

    # noinspection PyMethodMayBeStatic
    def show_result_train(self, data):
        plt.figure()
        plt.plot(data.history['accuracy'], label='accuracy')
        plt.plot(data.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
    # end show_train
