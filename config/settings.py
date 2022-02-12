# region python
import os
# endregion

# region package (third-party)
from environs import Env
# endregion


class Setting:
    __base_dir: str
    __env: Env

    def __init__(self):
        self.__base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__env = Env()
        self.__env.read_env()
    # end __init__()

    @property
    def base_dir(self) -> str:
        return self.__base_dir
    # end base_dir()

    @property
    def is_debug(self) -> bool:
        return self.__env.bool("DEBUG", default=True)
    # end is_debug()

    @property
    def logger_yaml(self) -> str:
        return os.path.join(self.__base_dir, 'config', 'logger.yaml')
    # end logger_yaml()

    @property
    def dataset_folder(self, dataset_type: str = 'colostate') -> str:
        return os.path.join(self.__base_dir, 'dataset', dataset_type)
    # end dataset_folder()

    @property
    def image_export_folder(self) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = os.path.join(self.__base_dir, 'exports')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Kiểm tra và khởi tạo thưc mục hình ảnh export
        folder = os.path.join(folder, 'images')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end image_export_folder()

    @property
    def training_folder(self) -> str:
        folder = self.image_export_folder

        # region Kiểm tra và khởi tạo thưc mục training
        folder = os.path.join(folder, 'training')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end training_folder()

    @property
    def validation_folder(self) -> str:
        folder = self.image_export_folder

        # region Kiểm tra và khởi tạo thưc mục training
        folder = os.path.join(folder, 'validation')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end validation_folder()

    @property
    def testing_folder(self) -> str:
        folder = self.image_export_folder

        # region Kiểm tra và khởi tạo thưc mục training
        folder = os.path.join(folder, 'testing')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder

    # end testing_folder()

    @property
    def h5_export(self) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = os.path.join(self.__base_dir, 'exports')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Kiểm tra và khởi tạo thưc mục H5 export
        folder = os.path.join(folder, 'h5')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end h5_export

    @property
    def evaluate_export_folder(self) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = os.path.join(self.__base_dir, 'evaluate')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end
# end Setting
