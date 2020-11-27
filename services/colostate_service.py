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
# enum
from models import EEGImageType
# models
from models import BrainComputerInterfaceModel as BCIModel, EEGSignalModel
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
        self.__target_channels = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'P3', 'P4']
    # end __init__()

    # region Private
    # region Load the Colorado State University dataset
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
                signal = np.array(signals[index], dtype=np.float)
                item = EEGSignalModel(name, channel, sample_rate, signal)
                result.append(item)
                self.__logger.debug('Kết thúc tải dữ liệu kênh: {0}'.format(channel))
            # end for

            self.__logger.debug('Kết thúc tải dữ liệu trial: {0}'.format(name))
        # end for

        self.__logger.debug('Kết thúc đọc dữ liệu các kênh')

        return result
    # end __load_eeg()
    # endregion

    # region Get image export path
    def __get_image_export_path(self, folder_name: str, channel: str, impairment: str, subject: int, device: str,
                                protocol: str, trial: str, is_full_time: bool, s_time: float, e_time: float) -> str:
        r"""
        Tạo path cho image
        :param folder_name: (time-series | spectrogram | scalogram)
        :param channel: Tên kênh đo
        :param impairment: Chuẩn đoán suy giảm thân kinh
        :param subject: Tên người thự nghiệm
        :param device: Thiết bị đo
        :param protocol: Cổng kết nối thiết bị
        :param trial: Các lần đo
        :param is_full_time: Hình ảnh thể hiện hết thời gian dữ liệu
        :param s_time: Thời gian bắt đầu
        :param e_time: Thời gian kết thúc
        :return: Path của image chuẩn bị export
        """
        # region Kiểm tra và khởi tạo thư mục loại hình
        if is_full_time:
            folder = f'{self.__setting.image_export_folder}\\full-time\\{folder_name}'
        else:
            folder = f'{self.__setting.image_export_folder}\\{folder_name}'
        # end if

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Khởi tạo thư mục channels
        folder = f'{folder}\\{channel}'

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Khởi tạo thư mục impairment
        impairment = impairment if impairment != 'none' else 'normal'
        folder = f'{folder}\\{impairment}'

        if not os.path.exists(folder):
            os.makedirs(folder)
        # end if
        # endregion

        # region Khởi tạo tên file
        s_time = str(s_time).replace('.', 's')
        e_time = str(e_time).replace('.', 's')
        trial = trial.replace(' ', '')
        file_name = f's{subject}_{device}_{protocol}_{trial}_{s_time}_{e_time}.png'
        # endregion

        return f'{folder}\\{file_name}'
    # end __get_image_export_path()

    def __get_time_series_image_export_path(self, channel: str, impairment: str, subject: int, device: str,
                                            protocol: str, trial: str, s_time: float, e_time: float,
                                            is_full_time: bool = False) -> str:
        r"""
        Tạo path cho image Time-Series
        :param channel: Tên kênh đo
        :param impairment: Chuẩn đoán suy giảm thân kinh
        :param subject: Tên người thự nghiệm
        :param device: Thiết bị đo
        :param protocol: Cổng kết nối thiết bị
        :param trial: Các lần đo
        :param s_time: Thời gian bắt đầu
        :param e_time: Thời gian kết thúc
        :param is_full_time: Hình ảnh thể hiện hết thời gian dữ liệu
        :return: Path của image Time-Series chuẩn bị export
        """
        return self.__get_image_export_path(
            folder_name=EEGImageType.TimeSeries.value,
            channel=channel,
            impairment=impairment,
            subject=subject,
            device=device,
            protocol=protocol,
            trial=trial,
            is_full_time=is_full_time,
            s_time=s_time,
            e_time=e_time
        )
    # end __get_time_series_image_export_path()

    def __get_spectrogram_image_export_path(self, channel: str, impairment: str, subject: int, device: str,
                                            protocol: str, trial: str, s_time: float, e_time: float,
                                            is_full_time: bool = False) -> str:
        r"""
        Tạo path cho image Spectrogram
        :param channel: Tên kênh đo
        :param impairment: Chuẩn đoán suy giảm thân kinh
        :param subject: Tên người thự nghiệm
        :param device: Thiết bị đo
        :param protocol: Cổng kết nối thiết bị
        :param trial: Các lần đo
        :param s_time: Thời gian bắt đầu
        :param e_time: Thời gian kết thúc
        :param is_full_time: Hình ảnh thể hiện hết thời gian dữ liệu
        :return: Path của image Spectrogram chuẩn bị export
        """
        return self.__get_image_export_path(
            folder_name=EEGImageType.Spectrogram.value,
            channel=channel,
            impairment=impairment,
            subject=subject,
            device=device,
            protocol=protocol,
            trial=trial,
            is_full_time=is_full_time,
            s_time=s_time,
            e_time=e_time
        )
    # end __get_time_series_image_export_path()

    def __get_scalogram_image_export_path(self, channel: str, impairment: str, subject: int, device: str, protocol: str,
                                          trial: str, s_time: float, e_time: float, is_full_time: bool = False) -> str:
        r"""
        Tạo path cho image Scalogram
        :param channel: Tên kênh đo
        :param impairment: Chuẩn đoán suy giảm thân kinh
        :param subject: Tên người thự nghiệm
        :param device: Thiết bị đo
        :param protocol: Cổng kết nối thiết bị
        :param trial: Các lần đo
        :param s_time: Thời gian bắt đầu
        :param e_time: Thời gian kết thúc
        :param is_full_time: Hình ảnh thể hiện hết thời gian dữ liệu
        :return: Path của image Scalogram chuẩn bị export
        """
        return self.__get_image_export_path(
            folder_name=EEGImageType.Scalogram.value,
            channel=channel,
            impairment=impairment,
            subject=subject,
            device=device,
            protocol=protocol,
            trial=trial,
            is_full_time=is_full_time,
            s_time=s_time,
            e_time=e_time
        )
    # end __get_time_series_image_export_path()
    # endregion

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
    def read_json(self, path: str) -> List[BCIModel]:
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
            eeg = item.get('eeg', dict())

            if isinstance(eeg, dict):
                eeg = self.__load_eeg(channels, sample_rate, eeg)
            # end if

            bci_model = BCIModel()
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

    def show_time_series(self, bci: List[BCIModel], bandpass: bool = False):
        for item in bci[:1]:
            for eeg in item.eeg[:1]:
                self.__eeg_service.show_time_series(eeg, bandpass)
            # end for
        # end for
    # end show_time_series()

    # region Export the image
    # region Export the time-series image
    def export_time_series(self, bci: List[BCIModel], bandpass: bool = False, segment_width: float = 3.84,
                           overlap: float = 0.2):
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
                index = 0
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)

                for segment in segments:
                    try:
                        s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                        e_time = np.round(s_time + segment_width, decimals=2)
                        path = self.__get_time_series_image_export_path(
                            channel=eeg.channel,
                            impairment=item.impairment,
                            subject=item.subject,
                            device=item.device,
                            protocol=item.protocol,
                            trial=eeg.trial,
                            s_time=s_time,
                            e_time=e_time
                        )
                        data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment)

                        # Export the time series of the signal
                        self.__eeg_service.export_time_series_image(path, data_export, bandpass)
                        index += 1
                    except Exception as ex:
                        self.__logger.error(ex)
                        continue
                    # end try
                # end for
            # end for
        # end for
    # end export_time_series()

    def export_full_time_series(self, bci: List[BCIModel], bandpass: bool = False):
        r"""
        Tạo ảnh Time Series full thời gian
        :param bci:
        :param bandpass:
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                try:
                    path = self.__get_time_series_image_export_path(
                        channel=eeg.channel,
                        impairment=item.impairment,
                        subject=item.subject,
                        device=item.device,
                        protocol=item.protocol,
                        trial=eeg.trial,
                        s_time=0,
                        e_time=len(eeg.signal),
                        is_full_time=True
                    )

                    # Export the time series of the signal
                    self.__eeg_service.export_time_series_image(path, eeg, bandpass, w=2581, h=1949)
                except Exception as ex:
                    self.__logger.error(ex)
                    continue
                # end try
            # end for
        # end for
    # end export_full_time_series()
    # endregion

    # region Export the spectrogram image
    def export_spectrogram(self, bci: List[BCIModel], segment_width: float = 3.84, overlap: float = 0.2):
        r"""
        Tạo ảnh Spectrogram
        :param bci:
        :param segment_width: Thời gian mỗi đoạn. Mặc định là 3.84s
        :param overlap: Tỉ lệ lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                index = 0
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)

                for segment in segments:
                    try:
                        s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                        e_time = np.round(s_time + segment_width, decimals=2)
                        path = self.__get_spectrogram_image_export_path(
                            channel=eeg.channel,
                            impairment=item.impairment,
                            subject=item.subject,
                            device=item.device,
                            protocol=item.protocol,
                            trial=eeg.trial,
                            s_time=s_time,
                            e_time=e_time
                        )
                        data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment)

                        # Export the time series of the signal
                        self.__eeg_service.export_spectrogram_image(path, data_export)
                        index += 1
                    except Exception as ex:
                        self.__logger.error(ex)
                        continue
                    # end try
                # end for
            # end for
        # end for
    # end export_spectrogram()

    def export_full_spectrogram(self, bci: List[BCIModel]):
        r"""
        Tạo ảnh Spectrogram full thời gian
        :param bci:
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                try:
                    path = self.__get_spectrogram_image_export_path(
                        channel=eeg.channel,
                        impairment=item.impairment,
                        subject=item.subject,
                        device=item.device,
                        protocol=item.protocol,
                        trial=eeg.trial,
                        s_time=0,
                        e_time=len(eeg.signal),
                        is_full_time=True
                    )

                    # Export the time series of the signal
                    self.__eeg_service.export_spectrogram_image(path, eeg, w=2581, h=1949)
                except Exception as ex:
                    self.__logger.error(ex)
                    continue
                # end try
            # end for
        # end for
    # end export_spectrogram()
    # endregion

    # region Export the scalogram image
    def export_scalogram(self, bci: List[BCIModel], segment_width: float = 3.84, overlap: float = 0.2):
        r"""
        Tạo ảnh Scalogram
        :param bci:
        :param segment_width: Thời gian mỗi đoạn. Mặc định là 3.84s
        :param overlap: Tỉ lệ lập của mỗi đoạn. Mặc định là 20%
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                index = 0
                # Số đoạn tính hiệu được chia cắt
                segments = self.__split_signal(signal=np.array(eeg.signal), sample_rate=eeg.sample_rate,
                                               segment_width=segment_width, overlap=overlap)

                for segment in segments:
                    try:
                        s_time = np.round(index * segment_width * (1 - overlap), decimals=2)
                        e_time = np.round(s_time + segment_width, decimals=2)
                        path = self.__get_scalogram_image_export_path(
                            channel=eeg.channel,
                            impairment=item.impairment,
                            subject=item.subject,
                            device=item.device,
                            protocol=item.protocol,
                            trial=eeg.trial,
                            s_time=s_time,
                            e_time=e_time
                        )
                        data_export = EEGSignalModel(eeg.trial, eeg.channel, eeg.sample_rate, segment)

                        # Export the time series of the signal
                        self.__eeg_service.export_scalogram_image(path, data_export)
                        index += 1
                    except Exception as ex:
                        self.__logger.error(ex)
                        continue
                    # end try
                # end for
            # end for
        # end for
    # end export_scalogram()

    def export_full_scalogram(self, bci: List[BCIModel]):
        r"""
        Tạo ảnh Scalogram full thời gian
        :param bci:
        :return:
        """
        for item in bci:
            for eeg in item.eeg:
                try:
                    path = self.__get_scalogram_image_export_path(
                        channel=eeg.channel,
                        impairment=item.impairment,
                        subject=item.subject,
                        device=item.device,
                        protocol=item.protocol,
                        trial=eeg.trial,
                        s_time=0,
                        e_time=len(eeg.signal),
                        is_full_time=True
                    )

                    # Export the time series of the signal
                    self.__eeg_service.export_scalogram_image(path, eeg, w=2581, h=1949)
                except Exception as ex:
                    self.__logger.error(ex)
                    continue
                # end try
            # end for
        # end for
    # end export_full_scalogram()
    # endregion

    def export_image(self, data_folder: str, eeg_image_type: EEGImageType):
        r"""
        Khởi tạo hình ảnh
        :param data_folder:
        :param eeg_image_type: (TimeSeries | Spectrogram| Scalogram | None)
        :return: Xuất ra các image (TimeSeries | Spectrogram| Scalogram | All)
        """
        if not data_folder:
            self.__logger.error('Thu muc dataset cua Colorado State University dang truyen vo rong')
            return
        # end if

        for (dir_path, dir_names, filenames) in walk(data_folder):
            for filename in filenames:
                # region Đọc dữ liệu từ file
                json_path = f'{dir_path}\\{filename}'
                data: List[BCIModel] = self.read_json(json_path)
                # endregion

                # region Export Image
                if eeg_image_type is None:
                    self.export_time_series(bci=data, bandpass=True)
                    self.export_spectrogram(bci=data)
                    self.export_scalogram(bci=data)
                # end if

                if eeg_image_type == EEGImageType.TimeSeries:
                    self.export_time_series(bci=data, bandpass=True)
                elif eeg_image_type == EEGImageType.Spectrogram:
                    self.export_spectrogram(bci=data)
                elif eeg_image_type == EEGImageType.Scalogram:
                    self.export_scalogram(bci=data)
                # end if
                # endregion
            # end for
        # end for
    # end export_image

    def export_full_time_image(self, data_folder: str, eeg_image_type: EEGImageType):
        r"""
        Khởi tạo hình ảnh full thời gian
        :param data_folder:
        :param eeg_image_type: (TimeSeries | Spectrogram| Scalogram | None)
        :return: Xuất ra các image (TimeSeries | Spectrogram| Scalogram | All)
        """
        if not data_folder:
            self.__logger.error('Thu muc dataset cua Colorado State University dang truyen vo rong')
            return
        # end if

        for (dir_path, dir_names, filenames) in walk(data_folder):
            for filename in filenames:
                # region Đọc dữ liệu từ file
                json_path = f'{dir_path}\\{filename}'
                data: List[BCIModel] = self.read_json(json_path)
                # endregion

                # region Export Image
                if eeg_image_type is None:
                    self.export_full_time_series(bci=data, bandpass=True)
                    self.export_full_spectrogram(bci=data)
                    self.export_full_scalogram(bci=data)
                # end if

                if eeg_image_type == EEGImageType.TimeSeries:
                    self.export_full_time_series(bci=data, bandpass=True)
                elif eeg_image_type == EEGImageType.Spectrogram:
                    self.export_full_spectrogram(bci=data)
                elif eeg_image_type == EEGImageType.Scalogram:
                    self.export_full_scalogram(bci=data)
                # end if
                # endregion
            # end for
        # end for
    # end export_image
    # endregion
    # endregion
# end ColostateService
