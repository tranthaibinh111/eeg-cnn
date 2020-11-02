# region python
import os
import pathlib
import shutil

from datetime import datetime
# endregion

# region package (third-party)
# region Numpy
import numpy as np
# endregion

# region tensorFlow
import tensorflow as tf
# endregion

# region PIL
from PIL import Image
# endregion
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# enum
from models import EEGChannel, Subject, ImpairmentType, EEGImageType, AIModelType
# services
from services import IocContainer, ColostateService, SimpleCNNService, AlexNetService, LeNetService
# endregion


def export(logger: Logger, setting: Setting, colostate_service: ColostateService, eeg_image_type: EEGImageType):
    r"""
    Thực hiện tạo dữ liệu hình ảnh cho việc trainning AI
    :param logger: ghi log.
    :param setting: cấu hình của ứng dụng.
    :param colostate_service:
    :param eeg_image_type: Thể loại hình ảnh sẻ tạo (TimeSeries | Spectrogram| Scalogram)
    :return:
    """
    try:
        # region Xuất hình ảnh tín hiệu bệnh
        subjects_impaired_folder = r'{0}\subjects_impaired'.format(setting.colostate_dataset)
        colostate_service.export_image(
            data_folder=subjects_impaired_folder,
            eeg_image_type=eeg_image_type
        )
        # endregion

        # region Xuất hình ảnh tín hiệu không bệnh
        subjects_not_impaired_folder = r'{0}\subjects_not_impaired'.format(setting.colostate_dataset)
        colostate_service.export_image(
            data_folder=subjects_not_impaired_folder,
            eeg_image_type=eeg_image_type
        )
        # endregion
    except Exception as ex:
        logger.error(ex)
    # end try
# end image_export()


def load_dataset(logger: Logger, data_folder: str, cnn_service):
    r"""
    Khởi tạo dữ liệu train (80%) / test (20%)
    :param logger: ghi log.
    :param data_folder: thư mục dataset
    :param cnn_service:
    :return:
    """
    try:
        # region Tạo dữ liệu train / test
        train_ds, val_ds = cnn_service.load_data(
            path=data_folder,
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


def evaluate_ai_model(logger: Logger, setting: Setting, eeg_image_type: EEGImageType, ai_model_type: AIModelType,
                      str_now: str):
    r"""
    Thực hiện đánh giá model training
    :param logger:
    :param setting:
    :param eeg_image_type: Loại hình ảnh đang được đánh giá
    :param ai_model_type:
    :param str_now: Thời gian mà model được khởi tạo - %Y%m%d%H%M%S
    :return:
    """
    try:
        # region Data test
        str_eeg_image_type = eeg_image_type.value
        logger.debug('Bat dau khoi tao data test')

        # region Lấy thông tin thư mục data và remove thư mục test data
        data_folder = r'{0}/{1}'.format(setting.colostate_image_export, str_eeg_image_type)
        test_folder = r'{0}/{1}'.format(setting.test_data_folder, str_eeg_image_type)

        # Kiểm tra nếu chưa tồn tại thư mục test data
        # Trường đã tồn tại thì xóa tất cả hình ảnh test
        # if not os.path.exists(test_folder):
        #     logger.debug('Khoi tao thu muc {0}"'.format(test_folder))
        #     os.makedirs(test_folder)
        # else:
        #     logger.debug('Remove tat ca file trong thu muc "{0}"'.format(test_folder))
        #     shutil.rmtree(test_folder)
        # end if
        # endregion

        # for channel in EEGChannel:
        #     logger.debug('Bat dau khoi tao data test voi channel "{0}"'.format(channel.value))
        #     # region Lấy thông tên channel và khởi tạo thư mục test channel
        #     str_channel = channel.value
        #     channel_path = r'{0}/{1}'.format(data_folder, str_channel)
        #     test_channel_path = r'{0}/{1}'.format(test_folder, str_channel)
        #
        #     if not os.path.exists(test_channel_path):
        #         logger.debug('Khoi tao thu muc "{0}"'.format(test_channel_path))
        #         os.makedirs(test_channel_path)
        #     # end if
        #     # endregion
        #
        #     for subject in Subject:
        #         # region Khởi tạo thư mục subject test
        #         str_subject = subject.value
        #         test_subject_folder = r'{0}/{1}'.format(test_channel_path, str_subject)
        #
        #         if not os.path.exists(test_subject_folder):
        #             message = 'Khoi tao thu muc "{0}"'.format(test_subject_folder)
        #             logger.debug(message)
        #             os.makedirs(test_subject_folder)
        #         # end if
        #         # endregion
        #
        #         # region Lấy các hình ảnh cho test
        #         data_dir = pathlib.Path(channel_path)
        #         image_filter = r'*/{0}_*.png'.format(str_subject)
        #         image_paths = list(data_dir.glob(image_filter))
        #         n_samples = int(np.ceil(len(image_paths) * 10e-2))
        #         index_random = np.random.randint(len(image_paths), size=n_samples)
        #         # endregion
        #
        #         # region Khởi tạo dữ liệu test
        #         logger.debug('Bat dau khoi tao data test "{0}" của {1}'.format(str_channel, str_subject))
        #         for index in index_random:
        #             file_path = image_paths[index]
        #
        #             if file_path:
        #                 logger.debug('Copy file {0} -> {1}'.format(file_path, test_subject_folder))
        #                 shutil.copy(file_path, test_subject_folder)
        #         # end for
        #         logger.debug('Ket thuc khoi tao data test "{0}" của {1}'.format(str_channel, str_subject))
        #         # endregion
        #     # end for
        #
        #     logger.debug('Ket thuc khoi tao data test voi channel "{0}"'.format(channel.value))
        # end for

        logger.debug('Ket thuc khoi tao data test')
        # endregion

        # region Đánh gia model AI
        str_ai_model_type = ai_model_type.value
        evaluation_matrix = np.zeros(shape=(13, 8), dtype=np.bool)
        col_index = 0

        for channel in EEGChannel:
            str_channel = channel.value
            logger.debug('Bat dau danh gia tren channel {0}'.format(str_channel))
            model_path = r'{0}\{1}\{2}_{3}_model_{4}.h5'.format(setting.h5_export, str_ai_model_type,
                                                                str_eeg_image_type, str_channel, str_now)
            cnn_model = tf.keras.models.load_model(model_path)
            row_index = 0

            for subject in Subject:
                str_subject = subject.value
                logger.debug('Bat dau nhan dang benh cho {0} dua tren channel {1}'.format(str_subject, str_channel))
                sub_folder = r'{0}/{1}/{2}'.format(test_folder, str_channel, str_subject)
                sub_dir = pathlib.Path(sub_folder)
                votes = np.array(list(), dtype=np.bool)

                for image_file in list(sub_dir.glob('*.png')):
                    logger.debug('Hinh anh: {0}'.format(str(image_file)))
                    # region Lấy matrix data image
                    image = Image.open(str(image_file))
                    image = image.convert('RGB')
                    image = image.resize(size=(270, 202))
                    data = np.array(image, dtype=float) / 255.
                    # Thêm thông sô batch (1, 270, 2002, 3)
                    data = np.expand_dims(data, axis=0)
                    # endregion

                    # region Thực hiện dự đoán
                    prediction = cnn_model.predict_classes(data, batch_size=1)

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

        for row in evaluation_matrix:
            row[-1] = np.sum(row[:6]) >= len(row) / 2
        # end for

        print(evaluation_matrix)
        # endregion
    except Exception as ex:
        logger.error(ex)
    # end try
# end evaluate_model


def main():
    # region Cấu hình chương trình
    # str_now: str = datetime.now().strftime('%Y%m%d%H%M%S')
    str_now: str = '20201029102707'
    eeg_image_type = EEGImageType.Spectrogram
    execute_export = False
    ai_model_type = AIModelType.LeNet
    execute_build_ai_model = False
    n_epochs = 50
    execute_evaluate = True
    # endregion

    # region Khai báo service
    container = IocContainer()
    # Utils
    logger: Logger = container.logger()
    # Configure
    setting: Setting = container.setting()
    # Services
    colostate_service: ColostateService = container.colostate_service()
    simple_cnn_service: SimpleCNNService = container.simple_cnn_service()
    alexnet_service: AlexNetService = container.alexnet_service()
    lenet_service: LeNetService = container.lenet_service()
    # endregion

    # region Export Image (Time Series | Spectrogram | Scalogram)
    if execute_export:
        export(logger, setting, colostate_service, eeg_image_type)
    # end if
    # endregion

    # region Build model AI và đánh giá
    if execute_build_ai_model:
        # region Khởi tạo service CNN
        if ai_model_type == AIModelType.SimpleCNN:
            cnn_service = simple_cnn_service
        elif ai_model_type == AIModelType.LeNet:
            cnn_service = lenet_service
        else:
            cnn_service = alexnet_service
        # end if
        # endregion

        for channel in [EEGChannel.P3, EEGChannel.P4]:
            # region Load dataset
            str_eeg_image_type: str = str(eeg_image_type.value)
            str_channel: str = str(channel.value).upper()
            data_folder: str = r'{0}\{1}\{2}'.format(setting.colostate_image_export, str_eeg_image_type, str_channel)
            train_ds, val_ds, labels = load_dataset(logger, data_folder, cnn_service)
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
        evaluate_ai_model(logger, setting, eeg_image_type, ai_model_type, str_now)
    # end if
    # endregion
# end main()


if __name__ == "__main__":
    main()
# end __name__
