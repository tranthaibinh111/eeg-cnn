# region python
from typing import Tuple
# endregion

# region package (third-party)
import numpy as np
import matplotlib.pyplot as plt
import pywt

from scipy.signal import butter, lfilter, stft
# endregion

# region motor impairment neural disorders
# utils
from utlis import Logger
# models
from models import EEGSignalModel
# endregion


class EEGService:
    # region Parameters
    __logger: Logger

    # endregion

    def __init__(self, logger: Logger):
        self.__logger = logger
    # end __init__()

    # region Private
    # noinspection PyMethodMayBeStatic
    def __butter_bandpass(self, low_cut: float, high_cut: float, fs: float,
                          order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html#butterworth-bandpass
        Butterworth digital and analog filter design.

        Design an Nth-order digital or analog Butterworth filter and return
        the filter coefficients.
        :param low_cut: The frequency min
        :param high_cut: The frequency max
        :param fs: The sample rate
        :param order: The order of the filter
        :return:
            b, a : ndarray, ndarray
            Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
            Only returned if ``output='ba'``.
        """
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(N=order, Wn=[low, high], btype='band')

        return b, a
    # end __butter_bandpass()

    # noinspection PyMethodMayBeStatic
    def __stft(self, fs: float, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute the Short Time Fourier Transform (STFT).
        [23] P. Xia, J. Hu, and Y. Peng, “UCI-Based Estimation of Limb Movement Using Deep Learning With Recurrent
        Convolutional Neural Networks,” Artif. Organs, vol. 42, no. 5, pp. E67–E77, 2018
        :param fs: The frequency
        :param data: ndarray (type: float)
            UCI signal data
        :return:
            frequencies: ndarray (type: float)
                Array of frequencies
            times: ndarray (type: float)
                Array of times
            power: ndarray (type: float)
                STFT of `x`
        """
        f, t, zxx = stft(
            data,
            fs=fs,
            window='hamming',
            nperseg=50,
            noverlap=20
        )

        # https://www.youtube.com/watch?v=g1_wcbGUcDY
        # see better (power in dBs)
        power = np.array(10 * np.ma.log10(np.abs(zxx)), dtype=np.float64)
        return f, t, power
    # end __stft()

    # noinspection PyMethodMayBeStatic
    def __cwt(self, data: np.ndarray) -> np.ndarray:
        r"""
        One dimensional Continuous Wavelet Transform.
        [25] M. P. G. Bhosale and S. T. Patil, “Classification of EEG Signals Using Wavelet Transform and Hybrid
        Classifier For Parkinson’s Disease Detection,” Int. J. Eng., vol. 2, no. 1, 2013.
        :param data: ndarray (type: float)
            UCI signal data
        :return:
            power: ndarray (type: float)
                WT of `x`
        """
        scales = np.arange(1, 51)
        coefs, _ = pywt.cwt(data, scales, 'morl')
        power = np.array(np.ma.log2(np.power(np.abs(coefs), 2)), dtype=np.float64)

        return power

    # end __cwt()
    # endregion

    # region Public
    # region Handle
    def butter_bandpass_filter(self, data: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 7.5,
                               order: int = 3) -> np.ndarray:
        r"""
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html#butterworth-bandpass
        We filtered frequency band from 0.5 Hz–7.5 Hz to remove low and high frequency noises and non-signal artefacts.
        Our primary focus was put on delta and theta waves frequency ranges, which based on previous work [2], [12]
        are containing spectral power changes which in most cases indicate some kind of brain pathologies.

        [2] V. Podgorelec, “Analyzing EEG signals with machine learning for diagnosing Alzheimer’s disease”,
            Elektronika ir Elektrotechnika, vol. 18, pp. 61–64, 2012. DOI: 10.5755/j01.eee.18.8.2627
        [12] R. Schirrmeister, L. Gemein, K. Eggensperger, F. Hutter, T. Ball, “Deep learning with convolutional neural
            networks for decoding and visualization of EEG pathology”, IEEE Signal Processing in Medicine and Biology
            Symposium, 2017. DOI: 10.1109/spmb.2017.8257015.
        :param data: An N-dimensional input array.
        :param lowcut: The frequency min
        :param highcut: The frequency max
        :param fs: The sample rate
        :param order: The order of the filter
        :return:
            y : array
            The output of the digital filter.
        """
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)

        return np.array(y, dtype=np.float)
    # end butter_bandpass_filter()

    def show_time_series(self, eeg_signal: EEGSignalModel, bandpass: bool = False):
        if not bandpass:
            signal = eeg_signal.signal
        else:
            signal = self.butter_bandpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
        # end if

        length = len(signal) / eeg_signal.sample_rate
        time = np.linspace(0, length, len(signal))

        plt.plot(time, signal)
        plt.show()
    # end show_time_series()

    def export_time_series_image(self, path: str, eeg_signal: EEGSignalModel, bandpass: bool = False, w: int = 496,
                                 h: int = 496, dpi: int = 300):
        r"""
        Export the time series image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :param bandpass: Are you execute the bandpass filter 0.5Hz - 7.5Hz
        :param w: image width
        :param h: image height
        :param dpi: dots-per-inch
        :return: The time series image of the signal
        """
        try:
            self.__logger.debug('Bắt đầu khởi tạo hình ảnh Time Series {0}'.format(path))
            signal = eeg_signal.signal if not bandpass else self.butter_bandpass_filter(eeg_signal.signal,
                                                                                        eeg_signal.sample_rate)
            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.axis('off')
            plt.plot(signal)
            plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(figure)
        except Exception as ex:
            self.__logger.error(ex)
        finally:
            self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Time Series {0}'.format(path))
        # end try
    # end export_time_series_image()

    def export_spectrogram_image(self, path: str, eeg_signal: EEGSignalModel, w: int = 496, h: int = 496,
                                 dpi: int = 300):
        r"""
        Export the spectrogram image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :param w: image width
        :param h: image height
        :param dpi: dots-per-inch
        :return: The spectrogram image of the signal
        """
        try:
            self.__logger.debug('Bắt đầu khởi tạo hình ảnh Spectrogram {0}'.format(path))
            signal = self.butter_bandpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
            f, t, power = self.__stft(eeg_signal.sample_rate, signal)
            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.axis('off')
            plt.pcolormesh(t.tolist(), f.tolist(), power.tolist(), cmap='viridis', vmin=np.amin(power),
                           vmax=np.amax(power), shading='auto')
            plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(figure)
        except Exception as ex:
            self.__logger.error(ex)
        finally:
            self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Spectrogram {0}'.format(path))
        # end try
    # end export_spectrogram_image()

    def export_scalogram_image(self, path: str, eeg_signal: EEGSignalModel, w: int = 496, h: int = 496, dpi: int = 300):
        r"""
        Export the scalogram image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :param w: image width
        :param h: image height
        :param dpi: dots-per-inch
        :return: The scalogram image of the signal
        """
        try:
            self.__logger.debug('Bắt đầu khởi tạo hình ảnh Scalogram {0}'.format(path))
            signal = self.butter_bandpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
            power = self.__cwt(signal)
            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.axis('off')
            plt.imshow(power.tolist(), cmap='jet', aspect='auto', vmin=np.amin(power), vmax=np.amax(power))
            plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(figure)
        except Exception as ex:
            self.__logger.error(ex)
        finally:
            self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Scalogram {0}'.format(path))
    # end export_scalogram_image()
    # endregion
    # endregion
# end EEGService
