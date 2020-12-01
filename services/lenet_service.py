# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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


class LeNetEarlyStopping(Callback):
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


class LeNetService(CNNService):
    def __init__(self,  setting: Setting, logger: Logger):
        super().__init__(setting, logger)

        self.img_width: int = 270
        self.img_height: int = 202
        self.batch_size: int = 1
        self.learning_rate: float = 10e-6

        self.model_folder: str = AIModelType.LeNet.value
        self.model_name: str = '{0}_model.h5'.format(AIModelType.LeNet.value)
        self.evaluate_folder: str = AIModelType.LeNet.value
        self.evaluate_name: str = '{0}_evaluate.png'.format(AIModelType.LeNet.value)
    # end __init__()

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

    def compile_and_fit_model(self, model, train_ds, val_ds, n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/lenet/lenet_model.h5"
        model_path = self.model_path(self.model_folder, self.model_name)

        # standardize
        train_ds = self.standardize_data(train_ds)
        val_ds = self.standardize_data(val_ds)

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
            ),
            LeNetEarlyStopping()
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
# end LeNetService
