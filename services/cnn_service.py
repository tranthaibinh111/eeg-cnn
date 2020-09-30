# region Python
import pathlib

from typing import List
# endregion

# region Package (third-party)
# region TensorFlow
import tensorflow as tf
# endregion

# region Seaborn
import seaborn as sns
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
    def get_data_and_labels(self, ds):
        normalized_ds = ds.map(lambda x, y: (x, y))
        data, labels = next(iter(normalized_ds))

        return data, labels
    # end get_data_and_labels()

    # noinspection PyMethodMayBeStatic
    def create_confusion_matrix(self, test_labels, pred_labels):
        confmat = confusion_matrix(y_true=test_labels, y_pred=pred_labels)

        return confmat
    # end create_confusion_matrix

    def show_confusion_matrix(self, labels, test_labels, pred_labels):
        confmat = self.create_confusion_matrix(test_labels, pred_labels)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

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
                ax.text(x=i, y=j, s=confmat[i, j], va='center', ha='center')

        # avoid that the first and last row cut in half
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()
        plt.show()
    # end show_confusion_matrix()

    def evaluate(self, test_labels: np.ndarray, pred_labels: np.ndarray):
        # region convert to binary array
        index = 0

        while index < len(test_labels):
            if test_labels[index] == pred_labels[index]:
                test_labels[index] = 1
                pred_labels[index] = 1
            else:
                test_labels[index] = 1
                pred_labels[index] = 0
            # end if

            index += 1
        # end while

        confmat = self.create_confusion_matrix(test_labels, pred_labels)
        tn, fp, fn, tp = confmat.ravel()

        # Sensitivity = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else .0
        self.__logger.debug("Sensitivity: %.2f%%" % (sensitivity * 100.0))

        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else .0
        self.__logger.debug("Specificity: %.2f%%" % (specificity * 100.0))

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        self.__logger.debug("Accuracy: %.2f%%" % (accuracy * 100.0))

        # F1 = 2TP / (2TP + FP + FN)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        self.__logger.debug("F1: %.2f%%" % (f1 * 100.0))

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        self.__logger.debug("Precision: %.2f%%" % (precision * 100.0))
    # end evaluate()
# end CNNService
