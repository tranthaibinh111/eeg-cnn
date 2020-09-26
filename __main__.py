# region Python
from typing import List
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


    # create cnn model
    cnn_model = __simple_cnn_service.build_model("relu", (img_height, img_width, 3))
    # train cnn model
    trained_cnn_model, cnn_history = __simple_cnn_service.compile_and_fit_model(cnn_model, train_ds, val_ds, n_epochs=50)
    # show result train
    __simple_cnn_service.show_result_train(cnn_history)
    # endregion
# end main()


if __name__ == "__main__":
    main()
# end __name__
