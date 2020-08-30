import os

from environs import Env


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
        return r'{0}\exports\images'.format(self.__base_dir)
    # end colostate_image_export()
# end Setting
