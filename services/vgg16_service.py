# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
# import keras (high level API) wiht tensorflow as backend
from tensorflow import data, keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
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

        x = vgg16_model.output
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(.5, name='fc1_drop')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(.5, name='fc2_drop')(x)
        predictions = Dense(output_class_units, activation='softmax', name='predictions')(x)

        model = tf.keras.Model(inputs=vgg16_model.input, outputs=predictions)
        # summarize the model
        model.summary()

        return model
    # end build_model()

    # noinspection PyMethodMayBeStatic
    def standardize_data(self, ds: data.Dataset) -> data.Dataset:
        r"""
        https://www.tensorflow.org/tutorials/images/transfer_learning#rescale_pixel_values
        :param ds: dataset
        :return: data standardized
        """
        ds = ds.map(lambda x, y: (preprocess_input(x), y))

        return ds
    # end get_data_and_labels()

    def compile_and_fit_model(self, model, train_ds, val_ds, n_epochs: int = 50):
        # Đường dẫn export file: "exports/h5/vgg16/vgg16_model.h5"
        model_path = self.model_path(self.model_folder, self.model_name)

        # standardize
        train_ds = self.standardize_data(train_ds)
        val_ds = self.standardize_data(val_ds)

        # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
        # compile the model
        model.compile(
            # optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True),
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
