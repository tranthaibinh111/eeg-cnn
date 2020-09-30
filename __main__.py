# region Python
from typing import List
# endregion

# region Package (third-party)
# region TensorFlow and tf.keras
import tensorflow as tf
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
# services
from services import ColostateService, SimpleCNNService, IocContainer
# models
from models import BrainComputerInterfaceModel
# endregion


def main():
    __container = IocContainer()

    # region Xuất dữ liệu cho việc train model
    __setting: Setting = __container.setting()
    # __logger: Logger = __container.logger()
    #
    # try:
    #     __colostate_service: ColostateService = __container.colostate_service()
    #
    #     # region Xuất hình ảnh tín hiệu bệnh
    #     subjects_impaired_folder = r'{0}\subjects_impaired'.format(__setting.colostate_dataset)
    #     __colostate_service.export_image(data_folder=subjects_impaired_folder)
    #     # endregion
    #
    #     # region Xuất hình ảnh tín hiệu không bệnh
    #     subjects_not_impaired_folder = r'{0}\subjects_not_impaired'.format(__setting.colostate_dataset)
    #     __colostate_service.export_image(data_folder=subjects_not_impaired_folder)
    #     # endregion
    # except Exception as e:
    #     __logger.error(e)
    # endregion

    # region Khởi tạo dữ liệu train / test
    __simple_cnn_service: SimpleCNNService = __container.simple_cnn_service()
    data_folder: str = r'{0}\scalograms\C3'.format(__setting.colostate_image_export)
    img_height = 180
    img_width = 180
    train_ds, val_ds = __simple_cnn_service.load_data(data_folder, img_height=img_height, img_width=img_width)
    labels = train_ds.class_names

    # Configure the dataset for performance
    autotune = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    val_data, val_labels = __simple_cnn_service.get_data_and_labels(val_ds)
    val_labels = val_labels.numpy()

    # create cnn model
    cnn_model = __simple_cnn_service.build_model("relu", (img_height, img_width, 3))

    # train cnn model
    cnn_model = __simple_cnn_service.compile_and_fit_model(cnn_model, train_ds, n_epochs=3)
    pred_labels = cnn_model.predict_classes(val_data)

    # evaluate
    __simple_cnn_service.evaluate(np.array(val_labels, copy=True), np.array(pred_labels, copy=True))

    # show confusion matrix
    __simple_cnn_service.show_confusion_matrix(labels, val_labels, pred_labels)
    # endregion
# end main()


if __name__ == "__main__":
    main()
# end __name__
