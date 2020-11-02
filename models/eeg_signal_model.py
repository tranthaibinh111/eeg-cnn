# region package (third-party)
# region Numpy
import numpy as np
# endregion
# endregion


class EEGSignalModel:
    # region Parameters
    # region Private
    __trial: str
    __channel: str
    __sample_rate: int
    __signal: np.ndarray
    # endregion
    # endregion

    # region GET
    @property
    def trial(self) -> str:
        return self.__trial

    # end channel()

    @property
    def channel(self) -> str:
        return self.__channel

    # end channel()

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    # end channel()

    @property
    def signal(self) -> np.ndarray:
        return self.__signal

    # end channel()
    # endregion

    def __init__(self, trial: str, channel: str, sample_rate: int, signal: np.ndarray):
        self.__trial = trial
        self.__channel = channel
        self.__sample_rate = sample_rate
        self.__signal = signal
    # end __init__()
# end EEGSignalModel
