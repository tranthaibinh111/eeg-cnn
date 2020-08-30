from typing import List

from .eeg_signal_model import EEGSignalModel


class BrainComputerInterfaceModel:
    # region Parameters
    # region Public
    protocol: str
    sample_rate: int
    notes: str
    channels: List[str]
    device: str
    location: str
    date: List[int]
    eeg: List[EEGSignalModel]
    impairment: str
    subject: int
    # endregion
    # endregion

    def __init__(self):
        self.target_indicator = None
        self.protocol = ''
        self.sample_rate = 0
        self.notes = ''
        self.channels = list()
        self.device = ''
        self. location = ''
        self.date = list()
        self.eeg = list()
        self.impairment = ''
        self.subject = 0
    # end __init__()
