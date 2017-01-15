#!/usr/bin/env python2

import sys
import math
import numpy as np
import librosa
import sklearn
import scipy.signal
from collections import Counter
# import filter


def highpass(freq):
    nyq = 0.5 * 22050
    return scipy.signal.butter(2, freq / nyq, 'highpass', False)


def lowpass(low, high):
    nyq = 0.5 * 22050
    return scipy.signal.butter(2, [low / nyq, high / nyq], 'bandpass', False)

# Seperate signal to x_low and x_high
def seperate_signal(sig):
    b, a = lowpass(70, 1000)
    x_low = scipy.signal.lfilter(b, a, sig)

    b, a = highpass(1000)
    x_high = scipy.signal.lfilter(b, a, sig) 
    # Half-wave rect.
    x_high = np.array([ (i if (i > 0) else 0) for i in x_high ])
    # Lowpass filter
    b, a = lowpass(70, 1000)
    x_high = scipy.signal.lfilter(b, a, x_high)
    return (x_low, x_high)


# Generate x2 by x_low and x_high.
def gen_x2(x_low, x_high):
    arr = np.square(np.fft.fft(x_low)) + np.square(np.fft.fft(x_high))
    return np.fft.ifft(arr)

# Return the peaks in x2.
def SACF_Enhancer(x2):
    # Select positive peaks
    peaks = librosa.util.peak_pick(x2, 8, 8, 16, 16, 0, 4)
    if len(peaks) == 0:
        return []

    peaks = peaks[np.where(x2[peaks] > 0)]
    # Select p in peaks where x[p] - x[p/2] > 0
    peaks = peaks[np.where((x2[peaks] - x2[peaks/2]) > 0)]
    return peaks

# Input should be 512 samples at 22050 Hz sampling rate (about 23ms).
# Return list of (freq, amplitude)
def gen_top3_freq(ip):
    x_low, x_high = seperate_signal(ip)
    x2 = gen_x2(x_low, x_high)

    x2 = np.abs(x2)
    peaks = SACF_Enhancer(x2)
    arr = list(peaks)
    arr.sort(key = lambda x: x2[x], reverse=True)
    arr = arr[:3]
    return [ ((22050.0/i), x2[i]) for i in arr ]

def extract_pitch_feature(y, sr):
    freq = []
    for i in xrange(0, len(y)-512, 256):
        freq += gen_top3_freq(y[i:i+512])
    # freq = [ gen_top3_freq(y[i:i+512]) for i in xrange(0, len(y)-512, 256) ])
    # print freq
    n = [ int(12 * math.log(f/440.0) / math.log(2) + 69) for f,_ in freq ]
    c = [ i % 12 for i in n ]
    cc = [ 7 * i % 12 for i in c ]
    # extract feature
    cnt = Counter()
    for f,a in freq:
        cnt[f] += a
    UP0 = 1.0 / cnt.most_common(1)[0][0]

    cnt.clear()
    for i in range(len(freq)):
        cnt[cc[i]] += freq[i][1]
    FA0 = cnt.most_common(1)[0][1]
    FP0 = 1.0 / cnt.most_common(1)[0][0]

    arr = cnt.most_common(2)
    IPO1 = arr[0][0] - arr[1][0]

    SUM = sum([ i for _,i in freq ])
    return [FA0, UP0, FP0, IPO1, SUM]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage : ./' + sys.argv[0] + ' <path>')
        exit(1)
    audio_path = sys.argv[1]
    y, sr = librosa.load(audio_path)
    f = extract_pitch_feature(y, sr)
    print(audio_path, tuple(f))
