# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
# endregion
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# enum
from enumerates import AIModelType
# services
from .cnn_service import CNNService
# endregion


class SimpleCNNCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # end if

        if logs.get("val_accuracy") == 1.00 and logs.get("val_loss") < 0.03:
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training = True
        # end if
    # end on_epoch_end()
# end SimpleCNNCallback


class SimpleCNNService(CNNService):
    def __init__(self, setting: Setting, logger: Logger):
        super().__init__(setting, logger)

        self.img_width: int = 384
        self.img_height: int = 384
        self.batch_size: int = 32

        self.model_folder: str = AIModelType.SimpleCNN.value
        self.model_name: str = '{0}_model.h5'.format(AIModelType.SimpleCNN.value)
        self.evaluate_folder: str = AIModelType.SimpleCNN.value
        self.evaluate_name: str = '{0}_evaluate.png'.format(AIModelType.SimpleCNN.value)
    # end __init__()

    # noinspection PyMethodMayBeStatic
    def build_model(self, output_class_units: int):
        r"""
        :param output_class_units:
        :return:
        """
        model = Sequential()

        # 3 Convolution layer with Max polling
        model.add(Conv2D(32, (5, 5), activation="relu", padding="same",
                         input_shape=(self.img_width, self.img_height, 3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (5, 5), activation="relu", padding="same", kernel_initializer="he_normal"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        # 3 Full connected layer
        model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
        model.add(Dense(54, activation="relu", kernel_initializer="he_normal"))
        model.add(Dense(output_class_units, activation="softmax"))  # 4 classes

        # summarize the model
        model.summary()

        return model
    # end build_model()

    # noinspection PyMethodMayBeStatic
    def compile_and_fit_model(self, model, train_ds, val_ds, n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/simple_cnn/simple_cnn_model.h5"
        model_path = self.model_path(self.model_folder, self.model_name)

        # Standardize the data
        train_ds = self.standardize_data(train_ds)
        val_ds = self.standardize_data(val_ds)

        # compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
            ),
            SimpleCNNCallback()
        ]

        # fit the model
        model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=self.batch_size,
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks)

        # Saving the model
        model = keras.models.load_model(model_path)

        return model
    # end compile_and_fit_model()
