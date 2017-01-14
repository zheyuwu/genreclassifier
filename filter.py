import scipy.signal
import numpy as np


SR = 22050


def highpass(freq):
    nyq = 0.5 * SR
    return scipy.signal.butter(2, freq / nyq, 'highpass', False)


def lowpass(low, high):
    nyq = 0.5 * SR
    return scipy.signal.butter(2, [low / nyq, high / nyq], 'bandpass', False)


if __name__ == '__main__':
    b, a = highpass(1000)
    sig = map(lambda t: np.sin(2 * np.pi * t * 16 / SR) * 10, range(SR))
    print(np.log10(np.mean(scipy.signal.lfilter(b, a, sig) ** 2)) * 10)
    sig = map(lambda t: np.sin(2 * np.pi * t * 32 / SR) * 10, range(SR))
    print(np.log10(np.mean(scipy.signal.lfilter(b, a, sig) ** 2)) * 10)

    b, a = lowpass(70, 1000)
    sig = map(lambda t: np.sin(2 * np.pi * t * 1024 / SR) * 10, range(SR))
    print(np.log10(np.mean(scipy.signal.lfilter(b, a, sig) ** 2)) * 10)
    sig = map(lambda t: np.sin(2 * np.pi * t * 2048 / SR) * 10, range(SR))
    print(np.log10(np.mean(scipy.signal.lfilter(b, a, sig) ** 2)) * 10)
