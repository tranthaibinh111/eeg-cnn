# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
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


class AlexNetEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # end if

        if logs.get("val_accuracy") == 1.00 and logs.get("val_loss") < 0.03:
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training = True
        # end if
    # end on_epoch_end()
# end EarlyStopping


class AlexNetService(CNNService):
    def __init__(self, setting: Setting, logger: Logger):
        super().__init__(setting, logger)

        self.img_width: int = 227
        self.img_height: int = 227
        self.batch_size: int = 1
        self.learning_rate: float = 10e-6

        self.model_folder: str = AIModelType.AlexNet.value
        self.model_name: str = '{0}_model.h5'.format(AIModelType.AlexNet.value)
        self.evaluate_folder: str = AIModelType.AlexNet.value
        self.evaluate_name: str = '{0}_evaluate.png'.format(AIModelType.AlexNet.value)
    # end __init__()

    # noinspection PyMethodMayBeStatic
    def build_model(self, output_class_units: int):
        r"""
        https://medium.com/analytics-vidhya/alexnet-tensorflow-2-1-0-d398b7c76cf
        :param output_class_units: Số class
        :return: AlexNet model
        """
        model = Sequential()

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu",
                         input_shape=(self.img_width, self.img_height, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd Layer: Conv (w ReLu)
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(BatchNormalization())

        # 4th Layer: Conv (w ReLu) splitted into two groups
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(BatchNormalization())

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        model.add(Flatten())

        # 7th Layer: FC (w ReLu) -> Dropout
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # 8th Layer: FC (w ReLu) -> Dropout
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(output_class_units, activation="softmax"))

        # summarize the model
        model.summary()

        return model
    # end build_model()

    def compile_and_fit_model(self, model, train_ds, val_ds, n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/alexnet/alexnet_model.h5"
        model_path = self.model_path(self.model_folder, self.model_name)

        # standardize
        train_ds = self.standardize_data(train_ds)
        val_ds = self.standardize_data(val_ds)

        # compile the model
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
            ),
            AlexNetEarlyStopping()
        ]

        # fit the model
        model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=self.batch_size,
            epochs=n_epochs,
            callbacks=callbacks)

        model = keras.models.load_model(model_path)

        return model
    # end compile_and_fit_model()
# end AlexNetService
