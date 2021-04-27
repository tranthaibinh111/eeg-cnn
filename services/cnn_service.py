# region Python
import pathlib
import os

from datetime import datetime
from typing import List, Tuple
# endregion

# region Package (third-party)
# region TensorFlow
from tensorflow import data
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# endregion

# region Scikit Learn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# endregion

# region Matplotlib
from matplotlib import pyplot as plt
# endregion

# region Numpy
import numpy as np
# endregion
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# endregion


class CNNService:
    # region Parameters
    setting: Setting
    logger: Logger

    img_height: int
    img_width: int
    batch_size: int
    learning_rate: float

    model_folder: str
    model_name: str
    evaluate_folder: str
    evaluate_name: str

    class_names: List[str]
    # endregion

    def __init__(self,  setting: Setting, logger: Logger):
        self.setting = setting
        self.logger = logger
    # end __init__()

    def load_data(self, training_folder: str, validation_folder: str, img_height: int, img_width: int,
                  batch_size: int) -> Tuple[data.Dataset, data.Dataset]:
        r"""
        Load dataset
        :param training_folder: thư mục data training
        :param validation_folder: thư mục data training
        :param img_height: Chiều cao hình ảnh
        :param img_width: Chiều rộng hình ảnh
        :param batch_size:
        :return:
            train_ds: dataset train
            val_ds: dataset test
        """
        # region Loading data folder
        training_dir = pathlib.Path(training_folder)
        validation_dir = pathlib.Path(validation_folder)

        n_training = len(list(training_dir.glob('*/*.png')))
        n_validation = len(list(training_dir.glob('*/*.png')))
        self.logger.debug(f'Tong data: {n_training + n_validation} images '
                          f'(training: {n_training} images | validation: {n_validation} images)')
        # endregion

        # region Create dataset
        train_ds: data.Dataset = image_dataset_from_directory(
            training_dir,
            image_size=(img_width, img_height),
            batch_size=batch_size)

        val_ds: data.Dataset = image_dataset_from_directory(
            validation_dir,
            image_size=(img_width, img_height),
            batch_size=batch_size)

        self.class_names = train_ds.class_names
        self.logger.debug(f'Class names: {self.class_names}')
        # endregion

        return train_ds, val_ds
    # end load_data()
    # endregion

    # region Build model
    def model_path(self, model_folder: str, model_name: str) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = r'{0}\{1}'.format(self.setting.h5_export, model_folder)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return r'{0}\{1}'.format(folder, model_name)
    # end model_path

    # noinspection PyMethodMayBeStatic
    def standardize_data(self, ds: data.Dataset) -> data.Dataset:
        r"""
        https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
        :param ds: dataset
        :return: data standardized
        """
        normalization_layer = Rescaling(1. / 255)

        ds = ds.map(lambda x, y: (normalization_layer(x), y))

        return ds
    # end get_data_and_labels()

    # noinspection PyMethodMayBeStatic
    def create_model_name(self, name: str) -> str:
        r"""
        Khởi tao tên model training kèm vơi thời gian
        :param name: Tên model muốn khởi tạo
        :return:
        """
        str_datetime = datetime.today().strftime('%Y%m%d%H%M%S')
        model_name = '{0}_{1}.h5'.format(name, str_datetime)

        return model_name
    # end create_model_name
    # endregion

    # region Evaluate model
    def evaluate_path(self, evaluate_folder: str, evaluate_name: str) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = r'{0}\{1}'.format(self.setting.evaluate_export_folder, evaluate_folder)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return r'{0}\{1}'.format(folder, evaluate_name)
    # endregion

    def show_confusion_matrix(self, labels, test_labels, pred_labels, path: str, show: bool):
        # region Confusion matrix
        confmat = confusion_matrix(y_true=test_labels, y_pred=pred_labels)
        self.logger.debug("Confusion matrix: {0}".format(confmat))
        # endregion

        # region Plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(confmat, cmap='Blues', alpha=0.5)

        n_labels = len(labels)
        ax.set_xticks(np.arange(n_labels))
        ax.set_yticks(np.arange(n_labels))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # loop over data dimensions and create text annotations.
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
            # end for
        # end for

        # avoid that the first and last row cut in half
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()

        if path:
            plt.savefig(fname=path, dpi=300.0, bbox_inches='tight', pad_inches=0)
        # end if

        if show:
            plt.show()
        # end if
        # endregion

        return confmat
    # end show_confusion_matrix()

    def evaluate(self, model, val_ds: data.Dataset, labels: List[str], show_evaluate: bool = True):
        # Đường dẫn export file: "evaluate/lenet/lenet_evaluate.h5"
        evaluate_path = self.evaluate_path(self.evaluate_folder, self.evaluate_name)

        val_ds = self.standardize_data(val_ds)
        val_labels = np.array([], dtype=np.int)
        pred_labels = np.array([], dtype=np.int)

        for image_data, label in list(val_ds):
            val_label = label.numpy()
            pred_label = model.predict_classes(image_data)
            val_labels = np.append(val_labels, val_label)
            pred_labels = np.append(pred_labels, pred_label)
        # end for

        # scikit-learn
        accuracy = accuracy_score(y_true=val_labels, y_pred=pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=val_labels, y_pred=pred_labels,
                                                                   average='weighted', zero_division=0)

        self.logger.debug("Sensitivity: {:.2f}%".format(recall * 100.0))
        self.logger.debug("Accuracy: {:.2f}%".format(accuracy * 100.0))
        self.logger.debug("F1: {:.2f}%".format(f1 * 100.0))
        self.logger.debug("Precision: {:.2f}%".format(precision * 100.0))

        # Show confusion matrix
        confmat = self.show_confusion_matrix(labels, val_labels, pred_labels, evaluate_path, show_evaluate)

        return confmat
    # end evaluate()
    # endregion
# end CNNService
