# region Python
from typing import List
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# services
from services import ColostateService, IocContainer
# models
from models import BrainComputerInterfaceModel
# endregion


def main():
    __container = IocContainer()
    __setting: Setting = __container.setting()
    __logger: Logger = __container.logger()

    try:
        __colostate_service: ColostateService = __container.colostate_service()
        json_path = r'{0}\subjects_impaired\{1}'.format(__setting.colostate_dataset, 's11-activetwo-home-impaired.json')
        data: List[BrainComputerInterfaceModel] = __colostate_service.read_json(json_path)
        __colostate_service.export_time_series(bci=data, bandpass=True)
        # __colostate_service.export_spectrogram(bci=data, impairment=True)
        # __colostate_service.export_scalogram(bci=data, impairment=True)
    except Exception as e:
        __logger.error(e)
# end main()


if __name__ == "__main__":
    main()
# end __name__
