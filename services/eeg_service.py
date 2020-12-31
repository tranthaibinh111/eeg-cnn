# region python
from typing import Tuple
# endregion

# region package (third-party)
import numpy as np
import matplotlib.pyplot as plt
import pywt

from scipy.signal import butter, filtfilt, stft
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
    def __butter_lowpass(self, cutoff: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Butterworth digital and analog filter design.

        Design an Nth-order digital or analog Butterworth filter and return
        the filter coefficients.
        :param cutoff: The frequency cut
        :param fs: The sample rate
        :param order: The order of the filter
        :return:
            b, a : ndarray, ndarray
            Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
            Only returned if ``output='ba'``.
        """
        nyq = 0.5 * fs
        normal = cutoff / nyq
        b, a = butter(N=order, Wn=normal, btype='lowpass', analog=False)

        return b, a
    # end __butter_lowpass()

    # noinspection PyMethodMayBeStatic
    def __butter_bandpass(self, low_cut: float, high_cut: float, fs: float,
                          order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
        b, a = butter(N=order, Wn=[low, high], btype='bandpass')

        return b, a
    # end __butter_bandpass()

    # noinspection PyMethodMayBeStatic
    def __stft(self, fs: float, data: np.ndarray, cutoff: float = 7.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute the Short Time Fourier Transform (STFT).
        [23] P. Xia, J. Hu, and Y. Peng, “UCI-Based Estimation of Limb Movement Using Deep Learning With Recurrent
        Convolutional Neural Networks,” Artif. Organs, vol. 42, no. 5, pp. E67–E77, 2018
        :param fs: The frequency
        :param data: ndarray (type: float)
            UCI signal data
        :param cutoff:
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

        # region Lọc lấy tần số 0 - 7.5
        f_cut = np.array(list(), dtype=np.float64)

        for y in f:
            f_cut = np.append(f_cut, y)

            if y >= cutoff:
                break
            # end if
        # end for

        zxx_cut = np.zeros(shape=(len(f_cut), len(t)), dtype=np.complex128)

        for y in range(len(f_cut)):
            for x in range(len(t)):
                zxx_cut[y, x] = zxx[y, x]
            # end for
        # end for
        # endregion

        # https://www.youtube.com/watch?v=g1_wcbGUcDY
        # see better (power in dBs)
        power = np.array(10 * np.ma.log10(np.abs(zxx_cut)), dtype=np.float64)

        return f_cut, t, power
    # end __stft()

    # noinspection PyMethodMayBeStatic
    def __cwt(self, data: np.ndarray, fs: float, cutoff: float = 7.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        One dimensional Continuous Wavelet Transform.
        [25] M. P. G. Bhosale and S. T. Patil, “Classification of EEG Signals Using Wavelet Transform and Hybrid
        Classifier For Parkinson’s Disease Detection,” Int. J. Eng., vol. 2, no. 1, 2013.
        :param data: ndarray (type: float)
            UCI signal data
        :param fs: the frequency
        :param cutoff:
        :return:
            power: ndarray (type: float)
                WT of `x`
        """
        scales = np.arange(1, fs + 1)
        sampling_period = 1 / fs
        coefs, freqs = pywt.cwt(data, scales, 'morl', sampling_period=sampling_period, method='fft')

        # region Lọc lấy tần số 0 - 7.5
        t = np.arange(0, coefs.shape[1])
        f_cut = np.array(list(), dtype=np.float64)

        for y in freqs:
            if y >= 7.5:
                continue
            # end if

            f_cut = np.append(f_cut, y)
        # end for

        coefs_cut = np.zeros(shape=(len(f_cut), len(t)), dtype=np.complex128)

        for y in range(len(f_cut)):
            for x in range(len(t)):
                coefs_cut[y, x] = coefs[y, x]
        # endregion

        power = np.array(np.ma.log2(np.power(np.abs(coefs_cut), 2)), dtype=np.float64)

        return f_cut, t, power

    # end __cwt()
    # endregion

    # region Public
    # region Handle
    def butter_lowpass_filter(self, data: np.ndarray, fs: float, cutoff: float = 7.5, order: int = 4) -> np.ndarray:
        r"""
        To test the proposed method we used the Colorado State University brain-computer (BCI) collection [19] of EEG
        signals, which were acquired using g.Tec g.GAMMASys active electrodes. Recordings where captured with eight
        active electrodes (8 channels) with sampling frequency of 256 Hz and a hardware bandpass filter from
        0.5 Hz –100 Hz at -3 dB attenuation [20]

        [20] E. Forney, C. Anderson, W. Gavin, et al., “Echo state networks for modeling and classification of EEG
            signals in mental-task braincomputer interfaces”, Colorado State University Technical Report CS-15-102, 2015
        :param data: An N-dimensional input array.
        :param cutoff: The frequency cut
        :param fs: The sample rate
        :param order: The order of the filter
        :return:
            y : array
            The output of the digital filter.
        """
        b, a = self.__butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)

        return np.array(y, dtype=np.float)
    # end butter_lowpass_filter()

    def butter_bandpass_filter(self, data: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 7.5,
                               order: int = 4) -> np.ndarray:
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
        y = filtfilt(b, a, data)

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

    def export_time_series(self, path: str, eeg_signal: EEGSignalModel, lowpass: bool = False, w: int = 496,
                           h: int = 496, dpi: int = 300):
        r"""
        Export the time series image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :param lowpass: Are you execute the bandpass filter 0.5Hz - 7.5Hz
        :param w: image width
        :param h: image height
        :param dpi: dots-per-inch
        :return: The time series image of the signal
        """
        try:
            self.__logger.debug('Bắt đầu khởi tạo hình ảnh Time Series {0}'.format(path))
            length = float(len(eeg_signal.signal)) / float(eeg_signal.sample_rate)
            time = np.linspace(0, length, len(eeg_signal.signal))

            if not lowpass:
                signal = eeg_signal.signal
            else:
                signal = self.butter_lowpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
            # end if

            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.plot(time, signal)
            plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(figure)
        except Exception as ex:
            self.__logger.error(ex)
        finally:
            self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Time Series {0}'.format(path))
        # end try
    # end export_time_series_image()

    def export_spectrogram(self, path: str, eeg_signal: EEGSignalModel, w: int = 496, h: int = 499, dpi: int = 300):
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
            signal = self.butter_lowpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
            f, t, power = self.__stft(eeg_signal.sample_rate, signal)

            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.axis('off')
            plt.pcolormesh(t.tolist(), f.tolist(), power.tolist(), cmap='viridis', vmin=np.amin(power),
                           vmax=np.amax(power), shading='gouraud')
            plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(figure)
        except Exception as ex:
            self.__logger.error(ex)
        finally:
            self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Spectrogram {0}'.format(path))
        # end try
    # end export_spectrogram_image()

    def export_scalogram(self, path: str, eeg_signal: EEGSignalModel, w: int = 496, h: int = 499, dpi: int = 300):
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
            signal = self.butter_lowpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
            f, t, power = self.__cwt(signal, eeg_signal.sample_rate)

            # Plot
            figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            plt.axis('off')
            plt.pcolormesh(t.tolist(), f.tolist(), power.tolist(), cmap='jet', vmin=np.amin(power),
                           vmax=np.amax(power), shading='gouraud')
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
