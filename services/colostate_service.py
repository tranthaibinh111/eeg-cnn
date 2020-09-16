# region python
import os

from os import walk
from typing import List
# endregion

# region package (third-party)
import numpy as np
import pandas
# endregion

# region motor impairment neural disorders
# config
from config import Setting
# utils
from utlis import Logger
# models
from models import BrainComputerInterfaceModel, EEGSignalModel
# service
from .eeg_service import EEGService
# endregion


class ColostateService:
    # region Parameters
    __setting: Setting
    __logger: Logger
    __eeg_service: EEGService
    __target_channels: List[str]
    # endregion

    def __init__(self, setting: Setting, logger: Logger, eeg_service: EEGService):
        self.__setting = setting
        self.__logger = logger
        self.__eeg_service = eeg_service
        self.__target_channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    # end __init__()

    # region Private
    def __check_struct_json(self, headers: List[str]) -> bool:
        self.__logger.debug('Bắt đầu kiểm tra cấu trúc file')

        if 'protocol' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns protocol')
            return False
        # end if

        if 'sample rate' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns sample rate')
            return False
        # end if

        if 'notes' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns notes')
            return False
        # end if

        if 'channels' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns channels')
            return False
        # end if

        if 'date' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns date')
            return False
        # end if

        if 'location' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns location')
            return False
        # end if

        if 'device' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns device')
            return False
        # end if

        if 'eeg' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns eeg')
            return False
        # end if

        if 'impairment' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns impairment')
            return False
        # end if

        if 'subject' not in headers:
            self.__logger.error('Cấu trúc file không chính xác. Không tìm thấy columns subject')
            return False
        # end if

        self.__logger.debug('Kết thúc kiểm tra cấu trúc file')

        return True
    # end __check_struct_json()

    def __load_eeg(self, channels: List[str], sample_rate: int, eeg: dict) -> List[EEGSignalModel]:
        """
        Load the EEG data
        :param channels: The array channels of device
        :param sample_rate: The frequency
        :param eeg: The EEG signal data
        :return:
            The channels data of eeg
        """
        self.__logger.debug('Bắt đầu đọc dữ liệu các kênh')
        result = list()

        for name, signals in eeg.items():
            self.__logger.debug('Băt đẩu tải dữ liệu trial: {0}'.format(name))

            for index in range(len(channels)):
                channel = channels[index]

                if not (channel in self.__target_channels):
                    continue
                # end if

                self.__logger.debug('Tải dữ liệu kênh: {0}'.format(channel))
                signal = signals[index]
                item = EEGSignalModel(name, channel, sample_rate, signal)
                result.append(item)
                self.__logger.debug('Kết thúc tải dữ liệu kênh: {0}'.format(channel))
            # end for

            self.__logger.debug('Kết thúc tải dữ liệu trial: {0}'.format(name))
        # end for

        self.__logger.debug('Kết thúc đọc dữ liệu các kênh')

        return result
    # end __load_eeg()

    def __image_export_path(self, image_type: str, impairment: str, subject: str, device: str, location: str,
                            channel: str, trial: str, s_time: float, e_time: float, extract: str) -> str:
        # region Kiểm tra và khởi tạo thư mục loại hình
        folder = r'{0}\{1}'.format(self.__setting.colostate_image_export, image_type)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Kiểm tra và khởi tạo thư mục hình là impaired / unimpaired
        is_disease = impairment != 'none'
        folder = r'{0}\{1}'.format(folder, 'impaired' if is_disease else 'unimpaired')

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if

        if is_disease:
            folder = r'{0}\{1}'.format(folder, impairment)

            if not os.path.exists(folder):
                os.makedirs(folder)
            # end if
        # end if
        # endregion

        # region Khởi tạo thư mục channels
        folder = r'{0}\{1}'.format(folder, channel)

        if not os.path.exists(folder):
            os.makedirs(folder)
        # endregion

        # region Khởi tạo tên file
        s_time = int(s_time) if s_time == int(s_time) else s_time
        e_time = int(e_time) if e_time == int(e_time) else e_time
        trial = trial.replace(' ', '')
        file_name = r'{0}-{1}-{2}-{3}-{4}-{5}'.format(subject, device, location, trial, s_time, e_time)
        # endregion

        return r'{0}\{1}.{2}'.format(folder, file_name, extract)
    # end __image_export_path()

    def __time_series_image_export_path(self, impairment: str, subject: str, device: str, location: str,
                                        channel: str, trial: str, s_time: float, e_time: float,
                                        image_type: str = 'time-series', extract: str = 'png') -> str:
        return self.__image_export_path(image_type, impairment, subject, device, location, channel, trial, s_time,
                                        e_time, extract)
    # end __time_series_image_export_path()

    def __spectrogram_image_export_path(self, impairment: str, subject: str, device: str, location: str,
                                        channel: str, trial: str, s_time: float, e_time: float,
                                        image_type: str = 'spectrograms', extract: str = 'png') -> str:
        return self.__image_export_path(image_type, impairment, subject, device, location, channel, trial, s_time,
                                        e_time, extract)
    # end __time_series_image_export_path()

    def __scalogram_image_export_path(self, impairment: str, subject: str, device: str, location: str, channel: str,
                                      trial: str, s_time: float, e_time: float, image_type: str = 'scalograms',
                                      extract: str = 'png') -> str:
        return self.__image_export_path(image_type, impairment, subject, device, location, channel, trial,
                                        s_time, e_time, extract)
    # end __time_series_image_export_path()

    # noinspection PyMethodMayBeStatic
    def __split_signal(self, signal: np.ndarray, sample_rate: int, segment_width: float = 3.84,
                       overlap: float = 0.2) -> np.ndarray:
        r"""
        Thực hiện chi tín hiệu thành từng đoạn
        :param signal: Dữ liệu tín hiệu
        :param sample_rate:
        :param segment_width: Thời gian của mỗi đoạn. Mặc định là 3.84s
        :param overlap: Phần trăm lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        signal_length = len(signal)
        segment_length = np.floor(segment_width * sample_rate)
        segment_overlap = np.floor(segment_length * overlap)
        index = 0
        segments = np.empty(0, dtype=float)

        while True:
            # region Tính toán ví trí đoạn cần cắt
            start = int(index * (segment_length - segment_overlap))
            end = int(start + segment_length)

            if end > signal_length:
                break
            # end if
            # endregion

            # region Thực hiện cắt đoạn
            segment = signal[start:end]

            if len(segment) == 0:
                break
            # end if

            if index < 1:
                segments = segment
            elif index == 1:
                segments = np.append([segments], [segment], axis=0)
            else:
                segments = np.append(segments, [segment], axis=0)
            # end if

            index += 1
            # endregion
        # end while

        return segments
    # end __split_signal()
    # endregion

    # region Public
    # noinspection PyMethodMayBeStatic
    def read_json(self, path: str) -> List[BrainComputerInterfaceModel]:
        """
        Load the brain coumput interface for the json file
        :param path: The file path
        :return:
            The brain computer interface list
        """
        self.__logger.debug('Bắt đầu đọc file json: {0}'.format(path))
        result = list()
        data = pandas.read_json(path, convert_dates=False)

        if not self.__check_struct_json(data.keys()):
            return list()
        # end if

        for index, item in data.iterrows():
            protocol = item.get('protocol', '')
            self.__logger.debug('Bắt đầu đọc protocol: {0}'.format(protocol))
            channels = item.get('channels', list())
            sample_rate = item.get('sample rate', 0)
            eeg = item.get('eeg', list())

            if isinstance(eeg, dict):
                eeg = self.__load_eeg(channels, sample_rate, eeg)
            # end if

            bci_model = BrainComputerInterfaceModel()
            bci_model.protocol = protocol
            bci_model.sample_rate = sample_rate
            bci_model.notes = item.get('notes', '')
            bci_model.channels = channels
            bci_model.device = item.get('device', '')
            bci_model.location = item.get('location', '')
            bci_model.date = item.get('date', list())
            bci_model.eeg = eeg
            bci_model.impairment = item.get('impairment', '')
            bci_model.subject = item.get('subject', 0)
            result.append(bci_model)
            self.__logger.debug('Kết thúc đọc protocol: {0}'.format(protocol))
        # end for

        self.__logger.debug('Kết thúc đọc file json: {0}'.format(path))

        return result
    # end read_jon()

    def show_time_series(self, bci: List[BrainComputerInterfaceModel], bandpass: bool = False):
        for item in bci[:1]:
            for eeg in item.eeg[:1]:
                self.__eeg_service.show_time_series(eeg, bandpass)
            # end for
        # end for
    # end show_time_series()

    def export_time_series(self, bci: List[BrainComputerInterfaceModel], bandpass: bool = False,
                           segment_width: float = 3.84, overlap: float = 0.2):
        r"""
        Tạo ảnh Time Series
        :param bci:
        :param bandpass:
        :param segment_width: Thời gian mỗi đoạn. Mặc định là 3.84s
        :param overlap: Tỉ lệ lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)
                index = 0

                for segment in segments:
                    s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                    e_time = np.round(s_time + segment_width, decimals=2)
                    # Export the time series of the signal
                    path = self.__time_series_image_export_path(item.impairment, item.subject, item.device,
                                                                item.location, eeg.channel, eeg.trial, s_time, e_time)
                    data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment.tolist())
                    self.__eeg_service.export_time_series(path, data_export, bandpass)
                    index += 1
                # end for
            # end for
        # end for
    # end export_time_series()

    def export_spectrogram(self, bci: List[BrainComputerInterfaceModel], segment_width: float = 3.84,
                           overlap: float = 0.2):
        r"""
        Tạo ảnh Spectrogram
        :param bci:
        :param segment_width: Thời gian mỗi đoạn. Mặc định là 3.84s
        :param overlap: Tỉ lệ lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)
                index = 0

                for segment in segments:
                    s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                    e_time = np.round(s_time + segment_width, decimals=2)
                    # Export the time series of the signal
                    path = self.__spectrogram_image_export_path(item.impairment, item.subject, item.device,
                                                                item.location, eeg.channel, eeg.trial, s_time, e_time)
                    data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment.tolist())
                    self.__eeg_service.export_spectrogram(path, data_export)
                    index += 1
                # end for
            # end for
        # end for
    # end export_spectrogram()

    def export_scalogram(self, bci: List[BrainComputerInterfaceModel], segment_width: float = 3.84,
                         overlap: float = 0.2):
        r"""
        Tạo ảnh Scalogram
        :param bci:
        :param segment_width: Thời gian mỗi đoạn. Mặc định là 3.84s
        :param overlap: Tỉ lệ lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)
                index = 0

                for segment in segments:
                    s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                    e_time = np.round(s_time + segment_width, decimals=2)
                    # Export the time series of the signal
                    path = self.__scalogram_image_export_path(item.impairment, item.subject, item.device, item.location,
                                                              eeg.channel, eeg.trial, s_time, e_time)
                    data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment.tolist())
                    self.__eeg_service.export_scalogram(path, data_export)
                    index += 1
                # end for
            # end for
        # end for
    # end export_scalogram()

    def export_image(self, data_folder: str, time_series: bool = True, spectrogram: bool = True,
                     scalogram: bool = True):
        if not data_folder:
            return
        # end if

        for (dirpath, dirnames, filenames) in walk(data_folder):
            for filename in filenames:
                # region Đọc dữ liệu từ file
                json_path = r'{0}\{1}'.format(dirpath, filename)
                data: List[BrainComputerInterfaceModel] = self.read_json(json_path)
                # endregion

                # region Export Image
                # Export Time Series Image
                if time_series:
                    self.export_time_series(bci=data, bandpass=True)
                # end if
                # Export Spectrogram Image
                if spectrogram:
                    self.export_spectrogram(bci=data)
                # end if
                # Export Scalogram
                if scalogram:
                    self.export_scalogram(bci=data)
                # end if
                # endregion
            # end for
        # end for
    # end export_image
    # endregion
# end ColostateService
