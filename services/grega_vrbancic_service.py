# region Python
import pathlib
import os
import re

from typing import List, Tuple
# endregion

# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
# endregion

# region Numpy
import numpy as np
# endregion

# region PIL
import PIL
# endregion
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# enum
from enumerates import ImpairmentType, Subject
# endregion


class GregaVrbancicService:
    # region Parameters
    setting: Setting
    logger: Logger

    model_folder: str
    model_name: str
    evaluate_folder: str
    evaluate_name: str
    # endregion

    def __init__(self,  setting: Setting, logger: Logger):
        self.img_width: int = 270
        self.img_height: int = 202
        self.learning_rate: float = 10e-6

        self.setting = setting
        self.logger = logger
        self.model_folder: str = 'grega-vrbancic'
        self.model_name: str = 'grega-vrbancic_model.h5'
        self.evaluate_folder: str = 'grega-vrbancic'
        self.evaluate_name: str = 'grega-vrbancic_evaluate.png'
    # end __init__()

    def load_data(self, data_folder: str, train_index: List[int]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Load dataset
        :param data_folder: thư mục data training
        :param train_index: danh sach thứ tự image được chọn để training
        :return:
            train_ds: dataset train
            val_ds: dataset test
        """
        # region Loading data folder
        data_dir = pathlib.Path(data_folder)
        paths = list(data_dir.glob('*/*.png'))
        n_data = len(paths)
        self.logger.debug(f'Tong data: {n_data} images (training: {len(train_index)} images | '
                          f'validation: {n_data - len(train_index)} images)')
        # endregion

        # region Create dataset
        training_x: np.ndarray = np.array(list(), dtype=np.int)
        training_y: np.ndarray = np.array(list(), dtype=np.int)
        validation_x: np.ndarray = np.array(list(), dtype=np.int)
        validation_y: np.ndarray = np.array(list(), dtype=np.int)

        for index in range(n_data):
            # region Lấy data image
            image_path = paths[index]
            image_pil = PIL.Image.open(image_path)
            image_pil = image_pil.convert(mode='RGB')
            x = np.array(image_pil.resize(size=(self.img_width, self.img_height)))
            # endregion

            # region Lấy thông tin class name
            subject_name = re.findall(r'^(\w\d\d)_.*\.png$', image_path.name)[0]
            impaired_ms = [Subject.S13.value, Subject.S15.value, Subject.S16.value]
            impaired_spinal = [Subject.S11.value]

            if subject_name in impaired_ms:
                y = ImpairmentType.MS.value
            elif subject_name in impaired_spinal:
                y = ImpairmentType.Spinal.value
            else:
                y = ImpairmentType.Normal.value
            # end if
            # endregion

            if index in train_index:
                if len(training_x) == 0:
                    training_x = np.array([x], dtype=np.int)
                else:
                    training_x = np.append(training_x, [x], axis=0)
                # end if
                training_y = np.append(training_y, y)
            else:
                if len(validation_x) == 0:
                    validation_x = np.array([x], dtype=np.int)
                else:
                    validation_x = np.append(validation_x, [x], axis=0)
                # end if
                validation_y = np.append(validation_y, y)
            # end if
        # end for
        # endregion

        return training_x, training_y, validation_x, validation_y
    # end load_data()

    def build_model(self, output_class_units: int):
        r"""
        https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086
        https://medium.com/@RaghavPrabhu/kaggles-digit-recogniser-using-tensorflow-lenet-architecture-92511e68cee1
        https://colab.research.google.com/drive/1kV3Jpxzup63GfJB1FGKxTSKd6Ek8J3sA#scrollTo=CAsRrp_nlP0y
        https://hackmd.io/@bouteille/S1WvJyqmI
        Input
            → Layer 1 → ReLu → Pooling
            → Layer 2 → ReLu → Pooling
            → FC1 → ReLu
            → FC2 → ReLu
            → FC3 → Yhat (using Softmax)
        :param output_class_units: Số class
        :return: LeNet model
        """
        model = Sequential()

        # 1st Layer: Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
        model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                         input_shape=(self.img_width, self.img_height, 3)))
        # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 2nd Layer: Convolutional. Output = 10x10x16.
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation="relu"))
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 3rd Layer: Flatten. Input = 5x5x16. Output = 400.
        model.add(Flatten())

        # 4th Layer: Fully Connected. Input = 400. Output = 120.
        model.add(Dense(120, activation='relu'))

        # 5th Layer: Fully Connected. Input = 120. Output = 84.
        model.add(Dense(84, activation='relu'))

        model.add(Dense(output_class_units, activation="softmax"))

        # summarize the model
        model.summary()

        return model
    # end build_model()

    def compile_and_fit_model(self, model, training_x, training_y, validation_x, validation_y,
                              n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/lenet/lenet_model.h5"
        # region Lấy thông tin path model
        model_folder = f'{self.setting.h5_export}\\{self.model_folder}'

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        # end if

        model_path = f'{model_folder}\\{self.model_name}'
        # endregion

        # standardize
        training_x = training_x / 255.
        validation_x = validation_x / 255.

        # compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
            )
        ]

        # fit the model
        model.fit(
            training_x,
            training_y,
            validation_data=(validation_x, validation_y),
            batch_size=1,
            epochs=n_epochs,
            callbacks=callbacks)

        model = keras.models.load_model(model_path)

        return model
    # end compile_and_fit_model()
# end GregaVrbancicService
