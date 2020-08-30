# region python
from typing import List, Tuple
# endregion

# region package (third-party)
import numpy as np
import matplotlib.pyplot as plt
import pywt

from dependency_injector import containers, providers
from scipy.signal import butter, freqz, lfilter, stft
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
    def __butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 3):
        r"""
        Butterworth digital and analog filter design.

        Design an Nth-order digital or analog Butterworth filter and return
        the filter coefficients.
        :param lowcut: The frequency min
        :param highcut: The frequency max
        :param fs: The sample rate
        :param order: The order of the filter
        :return:
            b, a : ndarray, ndarray
            Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
            Only returned if ``output='ba'``.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a
    # end __butter_bandpass()

    # noinspection PyMethodMayBeStatic
    def __stft(self, fs: float, data: List[float]) -> Tuple[List[float], List[float], List[float]]:
        r"""
        Compute the Short Time Fourier Transform (STFT).
        [23] P. Xia, J. Hu, and Y. Peng, “UCI-Based Estimation of Limb Movement Using Deep Learning With Recurrent Convolutional Neural Networks,” Artif. Organs, vol. 42, no. 5, pp. E67–E77, 2018
        :param fs: The frequency
        :param data: List[float]
            UCI signal data
        :return:
            frequencies: List[float]
                Array of frequencies
            times: List[float]
                Array of times
            power: List[float]
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
        power = np.array(np.ma.log10(zxx), dtype=np.float64)
        return f, t, power
    # end __stft()

    def __cwt(self, data: List[float]) -> List[float]:
        r"""
        One dimensional Continuous Wavelet Transform.
        [25] M. P. G. Bhosale and S. T. Patil, “Classification of EEG Signals Using Wavelet Transform and Hybrid Classifier For Parkinson’s Disease Detection,” Int. J. Eng., vol. 2, no. 1, 2013.
        :param data: List[float]
            UCI signal data
        :return:
            power: List[float]
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
    def butter_bandpass_filter(self, data: List[float], fs: float, lowcut: float = 0.5, highcut: float = 7.5,
                               order: int = 3):
        r"""
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
        return y
    # end butter_bandpass_filter()

    def export_time_series(self, path: str, eeg_signal: EEGSignalModel, bandpass: bool = False):
        r"""
        Export the time series image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :param bandpass: Are you execute the bandpass filter 0.5Hz - 7.5Hz
        :return: The time series image of the signal
        """
        self.__logger.debug('Bắt đầu khởi tạo hình ảnh Time Series {0}'.format(path))
        w = 496
        h = 499
        dpi = 300.0
        signal = eeg_signal.signal if not bandpass else self.butter_bandpass_filter(eeg_signal.signal,
                                                                                    eeg_signal.sample_rate)
        # length = np.ceil(len(signal) / float(eeg_signal.sample_rate))
        # time = np.linspace(0., length, num=int(length), endpoint=False)
        # Plot
        figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        plt.axis('off')
        plt.plot(signal)
        plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(figure)
        self.__logger.debug('Kết thúc quá trình khởi tạo hình ảnh Time Series {0}'.format(path))
    # end export_time_series()

    def export_spectrogram(self, path: str, eeg_signal: EEGSignalModel):
        r"""
        Export the spectrogram image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :return: The spectrogram image of the signal
        """
        w = 1291
        h = 499
        dpi = 300.0
        signal = self.butter_bandpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
        f, t, power = self.__stft(eeg_signal.sample_rate, signal)
        # Plot
        plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        plt.axis('off')
        plt.pcolormesh(t, f, power, cmap='viridis', vmin=power.min(), vmax=power.max())
        plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    # end export_time_series()

    def export_scalogram(self, path: str, eeg_signal: EEGSignalModel):
        r"""
        Export the scalogram image of the signal
        :param path: The path to save the file
        :param eeg_signal: The EEG signal
        :return: The scalogram image of the signal
        """
        w = 1291
        h = 499
        dpi = 300.0
        signal = self.butter_bandpass_filter(eeg_signal.signal, eeg_signal.sample_rate)
        power = self.__cwt(signal)
        # Plot
        plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(power, cmap='jet', aspect='auto', vmin=power.min(), vmax=power.max())
        plt.savefig(fname=path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    # end export_time_series()
    # endregion

    # region Show Plot
    def plot_butter_bandpass(self, fs: float, lowcut: float = 0.5, highcut: float = 7.5, orders: List[int] = [3, 6, 9]):
        r"""
        Plot the frequency response for a few different orders.
        :param fs: The sample rate
        :param lowcut: The frequency min
        :param highcut: The frequency max
        :param orders: The array order of the filter list
        """
        for order in orders:
            b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, np.abs(h), label="order = %d" % order)
        # end for

        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()
    # end plot_butter_bandpass()

    # def plot_mne(self):
    #     import mne
    #     from mne.io.edf.edf import RawEDF
    #
    #     eeg_file = '{0}\\dataset\\{1}'.format(os.getcwd(), 'huynh-van-phi.edf')
    #     channels = ['EEG F3-Cz', 'EEG F4-Cz', 'EEG C3-Cz', 'EEG C4-Cz', 'EEG P3-Cz', 'EEG P4-Cz', 'EEG O1-Cz', 'EEG O2-Cz']
    #     raw: RawEDF = mne.io.read_raw_edf(eeg_file, preload=True)
    #     print(len(raw.get_data('EEG F3-Cz')[0]))
    #     print(raw.info)
    #     raw.pick(channels).plot(duration=60, n_channels=8, scalings=dict(eeg=20e-5), block=True)
    # endregion
    # endregion
# end EEGService
