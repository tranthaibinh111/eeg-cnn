# region python
import pathlib
import re
import os

from datetime import datetime
from typing import List, Tuple
# endregion

# region package (third-party)
# region Numpy
import numpy as np
# endregion

# region tensorFlow
import tensorflow as tf

from tensorflow import data
# endregion

# region PIL
from PIL import Image
# endregion

# region sklearn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
# endregion

# region Matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
# endregion
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# enum
from enumerates import EEGChannel, Subject, ImpairmentType, EEGImageType, AIModelType
# services
from services import IocContainer, ColostateService, SimpleCNNService, LeNetService, AlexNetService, VGG16Service, \
    CNNService, GregaVrbancicService
# endregion


def export(logger: Logger, setting: Setting, colostate_service: ColostateService, eeg_image_type: EEGImageType,
           is_full_time: bool = False):
    r"""
    Thực hiện tạo dữ liệu hình ảnh cho việc trainning AI
    :param logger: ghi log.
    :param setting: cấu hình của ứng dụng.
    :param colostate_service:
    :param eeg_image_type: Thể loại hình ảnh sẻ tạo (TimeSeries | Spectrogram| Scalogram)
    :param is_full_time: export full thời gian
    :return:
    """
    try:
        # region Xuất hình ảnh tín hiệu bệnh
        subjects_impaired_folder = os.path.join(setting.dataset_folder, 'subjects_impaired')

        if is_full_time:
            colostate_service.export_full_time_image(
                data_folder=subjects_impaired_folder,
                eeg_image_type=eeg_image_type
            )
        else:
            colostate_service.export_image(
                data_folder=subjects_impaired_folder,
                eeg_image_type=eeg_image_type
            )
        # end if
        # endregion

        # region Xuất hình ảnh tín hiệu không bệnh
        subjects_not_impaired_folder = os.path.join(setting.dataset_folder, 'subjects_not_impaired')

        if is_full_time:
            colostate_service.export_full_time_image(
                data_folder=subjects_not_impaired_folder,
                eeg_image_type=eeg_image_type
            )
        else:
            colostate_service.export_image(
                data_folder=subjects_not_impaired_folder,
                eeg_image_type=eeg_image_type
            )
        # endregion
    except Exception as ex:
        logger.error(ex)
    # end try
# end image_export()


def load_dataset(logger: Logger, training_folder: str, validation_folder: str, cnn_service) \
        -> Tuple[data.Dataset, data.Dataset, List[str]]:
    r"""
    Khởi tạo dữ liệu train (80%) / test (20%)
    :param logger: ghi log.
    :param training_folder: thư mục data training
    :param validation_folder: thư mục data validation
    :param cnn_service:
    :return:
    """
    try:
        # region Tạo dữ liệu train / test
        train_ds, val_ds = cnn_service.load_data(
            training_folder=training_folder,
            validation_folder=validation_folder,
            img_height=cnn_service.img_height,
            img_width=cnn_service.img_width,
            batch_size=cnn_service.batch_size
        )
        labels = train_ds.class_names
        # endregion

        # region Configure the dataset for performance
        # https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
        autotune = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)
        # endregion

        return train_ds, val_ds, labels
    except Exception as ex:
        logger.error(ex)
    # end try


def build_ai_model(logger: Logger, cnn_service, train_ds, val_ds, n_labels, n_epochs: int) -> str:
    r"""
    Thực hiện tạo model training
    :param logger:
    :param cnn_service:
    :param train_ds: dữ liệu train
    :param val_ds: dữ liệu test
    :param n_labels: số lượng các lớp đầu ra
    :param n_epochs: số lần học
    :return:
    """
    try:
        # region Xây dựng model CNN
        # Khởi tạo model
        cnn_model = cnn_service.build_model(output_class_units=n_labels)
        # Thược hiện trainning
        cnn_service.compile_and_fit_model(cnn_model, train_ds, val_ds, n_epochs)
        # endregion

        return cnn_model
    except Exception as ex:
        logger.error(ex)
    # end try
# end build_model


def evaluate_ai_model(logger: Logger, setting: Setting, eeg_image_type: EEGImageType, service: CNNService,
                      str_now: str):
    r"""
    Thực hiện đánh giá model training
    :param logger:
    :param setting:
    :param eeg_image_type: Loại hình ảnh đang được đánh giá
    :param service:
    :param str_now: Thời gian mà model được khởi tạo - %Y%m%d%H%M%S
    :return:
    """
    try:
        # region Đánh gia model AI
        str_eeg_image_type = eeg_image_type.value
        test_folder = os.path.join(setting.testing_folder, str_eeg_image_type)
        model_folder = os.path.join(setting.h5_export, service.model_folder)

        evaluation_matrix = np.zeros(shape=(13, 9), dtype=np.bool)
        col_index = 0

        for channel in [EEGChannel.C3, EEGChannel.C4]:
            str_channel = channel.value
            logger.debug(f'Bat dau danh gia tren channel {str_channel}')
            model_file = f'{str_eeg_image_type}_{str_channel}_model_{str_now}.h5'
            model_path = os.path.join(model_folder, model_file)
            cnn_model = tf.keras.models.load_model(model_path)
            row_index = 0

            for subject in Subject:
                str_subject = subject.value
                logger.debug(f'Bat dau nhan dang benh cho {str_subject} dua tren channel {str_channel}')
                sub_folder = os.path.join(test_folder, str_channel, str_subject)
                sub_dir = pathlib.Path(sub_folder)
                votes = np.array(list(), dtype=np.bool)

                for image_file in list(sub_dir.glob('*.png')):
                    logger.debug('Hinh anh: {0}'.format(str(image_file)))
                    # region Lấy data image
                    image = Image.open(str(image_file))
                    image = image.convert('RGB')
                    image = image.resize(size=(service.img_width, service.img_height))
                    image_data = np.array(image, dtype=float) / 255.
                    # Thêm thông sô batch (1, 270, 2002, 3)
                    image_data = np.expand_dims(image_data, axis=0)
                    # endregion

                    # region Thực hiện dự đoán
                    prediction = cnn_model.predict_classes(image_data, batch_size=1)

                    if prediction[0]:
                        logger.debug('Index classifications: {0}'.format(prediction[0]))

                        if prediction[0] == ImpairmentType.Normal.value:
                            logger.debug('Nhan dang: UnImpairment'.format(prediction[0]))
                            votes = np.append(votes, False)
                        else:
                            logger.debug('Nhan dang: Impairment'.format(prediction[0]))
                            votes = np.append(votes, True)
                        # end if
                    # end if
                    # endregion
                # end for

                # region Bỏ phiếu đánh giá là Impairment hoặc UnImpairment
                number_vote_selected = np.sum(votes)
                sum_vote = len(votes)
                evaluation_matrix[row_index, col_index] = number_vote_selected >= sum_vote / 2.
                logger.debug('Bỏ phiếu đánh giá: {0}/{1}'.format(number_vote_selected, sum_vote))
                # endregion

                row_index = row_index + 1
                logger.debug('Ket thuc nhan dang benh cho {0} dua tren channel {1}'.format(str_subject, str_channel))
            # end for

            col_index = col_index + 1
            logger.debug('Ket thuc dau danh gia tren channel {0}'.format(str_channel))
        # end for

        logger.debug(str(evaluation_matrix))
        for row in evaluation_matrix:
            row[-1] = np.sum(row[:8]) >= (len(row) - 1) / 2
        # end for

        # region Ghi kết quả đánh giá model AI
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int)
        y_pred = np.array([row[-1] for row in evaluation_matrix], dtype=np.int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        logger.debug("Accuracy: {:.2f}%".format(accuracy * 100.0))

        sensitivity = tp / (tp + fn)
        logger.debug("Sensitivity: {:.2f}%".format(sensitivity * 100.0))

        specificity = tn / (tn + fp)
        logger.debug("Specificity: {:.2f}%".format(specificity * 100.0))

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.debug("F1: {:.2f}%".format(f1 * 100.0))
        # endregion
        # endregion

        # region Show Table plot
        y_labels = [item.value for item in Subject]
        x_labels = [item.value for item in EEGChannel]
        x_labels.append('class')

        fig, ax = plt.subplots(figsize=(8, 4))

        # hide axes
        ax.axis('off')
        ax.axis('tight')

        # Get some lists of color specs for row and column headers
        r_colors = cm.BuPu(np.full(len(y_labels), 0.1))
        c_colors = cm.BuPu(np.full(len(x_labels), 0.1))

        plt.table(
            cellText=evaluation_matrix,
            cellLoc='center',
            rowLabels=y_labels,
            rowColours=r_colors,
            rowLoc='center',
            colLabels=x_labels,
            colColours=c_colors,
            colLoc='center',
            loc='center left'
        )

        fig.patch.set_visible(False)
        fig.tight_layout()
        plt.show()
        # endregion
        # endregion
    except Exception as ex:
        logger.error(ex)
    # end try
# end evaluate_model


def main():
    # region Cấu hình chương trình
    str_now: str = datetime.now().strftime('%Y%m%d%H%M%S')
    # str_now: str = '20210109083800'
    eeg_image_type = EEGImageType.Scalogram
    execute_export = True
    is_full_time = False
    ai_model_type = AIModelType.AlexNet
    execute_build_ai_model = False
    n_epochs = 50
    execute_evaluate = False

    execute_grega_vrbancic = False
    # endregion

    # region Khai báo service
    container = IocContainer()
    # Utils
    logger: Logger = container.logger()
    # Configure
    setting: Setting = container.setting()
    # endregion

    # region Export Image (Time Series | Spectrogram | Scalogram)
    if execute_export:
        colostate_service: ColostateService = container.colostate_service()
        export(logger, setting, colostate_service, eeg_image_type, is_full_time)
    # end if
    # endregion

    if execute_grega_vrbancic:
        grega_vrbancic_service: GregaVrbancicService = container.grega_vrbancic_service()
        str_eeg_image_type: str = str(eeg_image_type.value)
        data_folder: str = os.path.join(setting.image_export_folder, 'full-time', 'str_eeg_image_type')

        # # region Training và validation LeNet
        # for channel in [EEGChannel.P3, EEGChannel.P4]:
        #     # region Lấy thông tin hình ảnh của channel
        #     str_channel: str = str(channel.value).upper()
        #     data_dir = pathlib.Path(f'{data_folder}\\{str_channel}')
        #     paths = list(data_dir.glob('*/*.png'))
        #     # endregion
        #
        #     # Tạo model LeNet
        #     class_names = [ImpairmentType.MS, ImpairmentType.Normal, ImpairmentType.Spinal]
        #     lenet_model = grega_vrbancic_service.build_model(len(class_names))
        #
        #     # region Thực hiện Leave-one-out
        #     loo = LeaveOneOut()
        #
        #     for train_index, test_index in loo.split(paths):
        #         # Load data
        #         training_x, training_y, validation_x, validation_y = grega_vrbancic_service.load_data(
        #             f'{data_folder}\\{str_channel}',
        #             train_index
        #         )
        #         # Compile and fit model
        #         grega_vrbancic_service.model_name = f'{str_eeg_image_type}_{str_channel}_model_{str_now}.h5'
        #         lenet_model = grega_vrbancic_service.compile_and_fit_model(lenet_model, training_x, training_y,
        #                                                                    validation_x, validation_y)
        #     # end for
        #     # endregion
        # # end for
        # # endregion

        # region Đánh giá mode LeNet
        img_width = grega_vrbancic_service.img_width
        img_height = grega_vrbancic_service.img_height
        test_folder = data_folder
        evaluation_matrix = np.zeros(shape=(13, 9), dtype=np.bool)
        col_index = 0

        # region Thực hiện đánh giá và lưu vào bẳng ma trận
        for channel in EEGChannel:
            # region Lấy thông tin hình ảnh của channel
            str_channel: str = str(channel.value).upper()
            data_dir = pathlib.Path(os.path.join(test_folder, str_channel))
            paths = list(data_dir.glob('*/*.png'))
            # endregion

            # region Load model của channel
            lenet_model_name: str = f'{str_eeg_image_type}_{str_channel}_model_{str_now}.h5'
            lenet_model_path: str = os.path.join(setting.h5_export, grega_vrbancic_service.model_folder, lenet_model_name)
            lenet_model = tf.keras.models.load_model(lenet_model_path)
            # endregion

            for index in range(len(paths)):
                # region Load data image
                image_path = paths[index]
                image_pil = Image.open(image_path)
                image_pil = image_pil.convert(mode='RGB')
                image_pil = image_pil.resize((img_width, img_height))
                x = np.array(image_pil)
                # endregion

                # standardize
                x = x / 255.
                # Thêm thông sô batch (1, 270, 2002, 3)
                x = np.expand_dims(x, axis=0)

                # region Lấy thông tin class name
                subject_name = re.findall(r'^(\w\d\d)_.*\.png$', image_path.name)[0]
                # impaired_ms = [Subject.S13.value, Subject.S15.value, Subject.S16.value]
                # impaired_spinal = [Subject.S11.value]
                #
                # if subject_name in impaired_ms:
                #     y_true = ImpairmentType.MS
                # elif subject_name in impaired_spinal:
                #     y_true = ImpairmentType.Spinal
                # else:
                #     y_true = ImpairmentType.Normal
                # # end if
                # endregion

                # Thực hiện dự đoán
                prediction = lenet_model.predict_classes(x, batch_size=1)[0]

                logger.debug(f'Image: {image_path.name}')
                logger.debug(f'Prediction: {prediction}')

                # region Cập nhât kêt quả dự đoán vào bảng đánh giá
                row_index = 0

                for subject in Subject:
                    if subject_name == subject.value:
                        break
                    # end if

                    row_index = row_index + 1
                # end for

                # evaluation_matrix[row_index, col_index] = y_true == prediction
                if prediction == ImpairmentType.Normal.value:
                    evaluation_matrix[row_index, col_index] = False
                else:
                    evaluation_matrix[row_index, col_index] = True
                # end if
                # endregion
            # end for

            col_index = col_index + 1
        # end for
        # endregion

        # region Tính toán accuracy, sensitivity, specificity, f1 từ ma trận
        logger.debug(str(evaluation_matrix))
        for row in evaluation_matrix:
            row[-1] = np.sum(row[:7]) >= (len(row) - 1) / 2
        # end for

        # region Ghi kết quả đánh giá model AI
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int)
        y_pred = np.array([row[-1] for row in evaluation_matrix], dtype=np.int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        logger.debug("Accuracy: {:.2f}%".format(accuracy * 100.0))

        sensitivity = tp / (tp + fn)
        logger.debug("Sensitivity: {:.2f}%".format(sensitivity * 100.0))

        specificity = tn / (tn + fp)
        logger.debug("Specificity: {:.2f}%".format(specificity * 100.0))

        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.debug("F1: {:.2f}%".format(f1 * 100.0))
        # endregion
        # endregion

        # region Show Table plot
        y_labels = [item.value for item in Subject]
        x_labels = [item.value for item in EEGChannel]
        x_labels.append('class')

        fig, ax = plt.subplots(figsize=(8, 4))

        # hide axes
        ax.axis('off')
        ax.axis('tight')

        # Get some lists of color specs for row and column headers
        r_colors = cm.BuPu(np.full(len(y_labels), 0.1))
        c_colors = cm.BuPu(np.full(len(x_labels), 0.1))

        plt.table(
            cellText=evaluation_matrix,
            cellLoc='center',
            rowLabels=y_labels,
            rowColours=r_colors,
            rowLoc='center',
            colLabels=x_labels,
            colColours=c_colors,
            colLoc='center',
            loc='center left'
        )

        fig.patch.set_visible(False)
        fig.tight_layout()
        plt.show()
        # endregion
        # endregion
    # end if

    if not execute_build_ai_model and not execute_evaluate:
        return
    # end if

    # region Khởi tạo service CNN
    if ai_model_type == AIModelType.SimpleCNN:
        cnn_service = container.simple_cnn_service()
    elif ai_model_type == AIModelType.LeNet:
        cnn_service = container.lenet_service()
    elif ai_model_type == AIModelType.AlexNet:
        cnn_service = container.alexnet_service()
    elif ai_model_type == AIModelType.VGG16:
        cnn_service = container.vgg16_service()
    # end if
    # endregion

    # region Build model AI và đánh giá
    if execute_build_ai_model:
        for channel in [EEGChannel.C3, EEGChannel.C4]:
            # region Load dataset
            str_eeg_image_type: str = str(eeg_image_type.value)
            str_channel: str = str(channel.value).upper()
            training_folder: str = os.path.join(setting.training_folder, str_eeg_image_type, str_channel)
            validation_folder: str = os.path.join(setting.validation_folder, str_eeg_image_type, str_channel)
            train_ds, val_ds, labels = load_dataset(logger, training_folder, validation_folder, cnn_service)
            # endregion

            # region Xây dựng model CNN
            cnn_service.model_name = '{0}_{1}_model_{2}.h5'.format(str_eeg_image_type, str_channel, str_now)
            cnn_model = build_ai_model(logger, cnn_service, train_ds, val_ds, len(labels), n_epochs)
            # endregion

            # region Đánh giá về model AI
            cnn_service.evaluate_name = r'{0}_{1}_evaluate_{2}.png'.format(str_eeg_image_type, str_channel, str_now)
            cnn_service.evaluate(cnn_model, val_ds, labels, show_evaluate=False)
            # end if
            # endregion
        # end for
    # end if
    # endregion

    # region Thực hiện đánh giá theo bài báo
    if execute_evaluate:
        evaluate_ai_model(logger, setting, eeg_image_type, cnn_service, str_now)
    # end if
    # endregion
# end main()


if __name__ == "__main__":
    main()
# end __name__
