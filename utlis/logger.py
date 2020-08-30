# region python
import logging
import logging.config
# endregion

# region package (third-party)
import yaml
# endregion

# region motor impairment neural disorders
from config import Setting
# endregion


class Logger:
    __logger = None

    def __init__(self, setting: Setting):
        self.__config(setting.logger_yaml)
    # end __init__()

    # region Private
    def __config(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

        self.__logger = logging.getLogger()
    # end __config()
    # endregion

    def debug(self, message: str):
        self.__logger.debug(message)
    # end debug()

    def error(self, ex: any):
        self.__logger.error(ex)
    # end error()
# end Logger
