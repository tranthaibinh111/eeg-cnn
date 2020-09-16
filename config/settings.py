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
        return r'{0}\config\logger.yaml'.format(self.__base_dir)
    # end logger_yaml()

    @property
    def colostate_dataset(self) -> str:
        return r'{0}\dataset\colostate'.format(self.__base_dir)
    # end colostate_dataset()

    @property
    def colostate_image_export(self) -> str:
        # region Kiểm tra và khởi tạo thư mục export
        folder = r'{0}\exports'.format(self.__base_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # endregion

        # region Kiểm tra và khởi tạo thưc mục hình ảnh export
        folder = r'{0}\images'.format(folder)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        return folder
    # end colostate_image_export()
# end Setting
