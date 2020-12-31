# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
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


class VGG16EarlyStopping(Callback):
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


class VGG16Service(CNNService):
    def __init__(self, setting: Setting, logger: Logger):
        super().__init__(setting, logger)

        self.img_width: int = 224
        self.img_height: int = 224
        self.batch_size: int = 1
        # self.learning_rate: float = 10e-4
        self.learning_rate: float = 10e-6

        self.model_folder: str = AIModelType.VGG16.value
        self.model_name: str = '{0}_model.h5'.format(AIModelType.VGG16.value)
        self.evaluate_folder: str = AIModelType.VGG16.value
        self.evaluate_name: str = '{0}_evaluate.png'.format(AIModelType.VGG16.value)
    # end __init__()

    # noinspection PyMethodMayBeStatic
    def build_model(self, output_class_units: int):
        r"""
        https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
        :param output_class_units: Số class
        :return: VGG16 model
        """
        vgg16_model = VGG16(
            include_top=False,
            weights=None,
            input_shape=(self.img_width, self.img_height, 3)
        )
        vgg16_model.trainable = False
        vgg16_model.summary()
        print('VGG Pretrained Model loaded.')

        model = Sequential()
        model.add(vgg16_model)
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        # model.add(Dropout(.5, name='fc1_drop'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        # model.add(Dropout(.5, name='fc2_drop'))
        model.add(Dense(output_class_units, activation='softmax', name='predictions'))

        # summarize the model
        model.summary()

        return model
    # end build_model()

    def compile_and_fit_model(self, model, train_ds, val_ds, n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/vgg16/vgg16_model.h5"
        model_path = self.model_path(self.model_folder, self.model_name)

        # standardize
        train_ds = self.standardize_data(train_ds)
        val_ds = self.standardize_data(val_ds)

        # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
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
            VGG16EarlyStopping()
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
