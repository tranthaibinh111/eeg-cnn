# region package (third-party)
import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# services
from services import EEGService, ColostateService, CNNService
# endregion


class IocContainer(containers.DeclarativeContainer):
    setting = providers.Singleton(Setting)
    logger = providers.Singleton(Logger, setting=setting)

    # services
    eeg_service = providers.Factory(EEGService, logger=logger)
    colostate_service = providers.Factory(ColostateService, setting=setting, logger=logger, eeg_service=eeg_service)
    cnn_service = providers.Factory(CNNService, logger=logger)
