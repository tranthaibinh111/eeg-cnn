# region python
from enum import Enum
# endregion


class EEGImageType(Enum):
    TimeSeries = 'time-series'
    Spectrogram = 'spectrogram'
    Scalogram = 'scalogram'
# end ImageType
