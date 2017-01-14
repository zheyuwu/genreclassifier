#!/usr/bin/env python2

import sys
import math
import numpy as np
import librosa
import sklearn
import scipy.signal
from collections import Counter

def extract_timbral_texture_feature(y, sr, texture_window, analysis_window):
    tw = texture_window
    aw = analysis_window
    M = np.absolute(librosa.core.stft(y=y, n_fft=tw))

    # Spectral Centroid
    spectral_centroid = np.array([ i.dot(xrange(1,len(i)+1)) / np.sum(i) for i in M ])

    #Spectral Rolloff
    vfunc = np.vectorize(np.sum)
    presum = np.array([ np.cumsum(i) for i in M ])
    spectral_rolloff = np.array([ np.searchsorted(arr, arr[-1]*0.85)+1 for arr in presum ])

    # Spectral Flux
    norm_M = np.array([ librosa.util.normalize(i) for i in M ])
    spectral_flux = np.array([ np.sum(np.square(i)) for i in (norm_M[1:] - norm_M[:-1]) ])

    # Time domain zero crossings
    z = np.cumsum(librosa.zero_crossings(y))
    zero_crossings = (z[tw:] - z[:-tw]) * 0.5

    # Low-Energy Feature
    presum = np.cumsum(np.square(y))
    avg = np.average(np.sqrt((presum[tw:] - presum[:-tw]) / tw))
    x = np.sqrt((presum[aw:] - presum[:-aw]) / aw)
    low_energy_ratio = 1.0 * len(x[np.where(x < avg)]) / len(x)

    # The first five MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)

    items = [spectral_centroid, spectral_rolloff, spectral_flux, zero_crossings]

    res = [ low_energy_ratio ]
    for i in items:
        res.append(i.mean())
        res.append(i.var())
    for i in mfcc:
        res.append(i.mean())
        res.append(i.var())
    return res

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
    assert len(peaks) > 0

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

# audio_path = 'genres/hiphop/hiphop.00000.au'
if len(sys.argv) != 2:
    print('usage : ./' + sys.argv[0] + ' <path>')
    exit(1)
audio_path = sys.argv[1]

y, sr = librosa.load(audio_path)

feature = []
f = extract_timbral_texture_feature(y, sr, 512 * 43, 512)
assert len(f) == 19
feature += f

f = extract_pitch_feature(y, sr)
assert len(f) == 5
feature += f

print(audio_path, tuple(feature))
