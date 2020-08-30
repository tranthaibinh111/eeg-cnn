from typing import List


class EEGSignalModel:
    # region Parameters
    # region Private
    __trial: int
    __channel: str
    __sample_rate: int
    __signal: List[float]
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
    def signal(self) -> List[float]:
        return self.__signal

    # end channel()
    # endregion

    def __init__(self, trial: str, channel: str, sample_rate: int, signal: List[float]):
        self.__trial = trial
        self.__channel = channel
        self.__sample_rate = sample_rate
        self.__signal = signal
    # end __init__()
# end EEGSignalModel
