# region package (third-party)
import dependency_injector.containers as containers
import dependency_injector.providers as providers
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# services
from services import EEGService, ColostateService, SimpleCNNService, AlexNetService, LeNetService, GregaVrbancicService
# endregion


class IocContainer(containers.DeclarativeContainer):
    setting = providers.Singleton(Setting)
    logger = providers.Singleton(Logger, setting=setting)

    # services
    eeg_service = providers.Factory(EEGService, logger=logger)
    colostate_service = providers.Factory(ColostateService, setting=setting, logger=logger, eeg_service=eeg_service)
    simple_cnn_service = providers.Factory(SimpleCNNService, setting=setting, logger=logger)
    alexnet_service = providers.Factory(AlexNetService, setting=setting, logger=logger)
    lenet_service = providers.Factory(LeNetService, setting=setting, logger=logger)

    grega_vrbancic_service = providers.Factory(GregaVrbancicService, setting=setting, logger=logger)
