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

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage : ./' + sys.argv[0] + ' <path>')
        exit(1)
    audio_path = sys.argv[1]
    y, sr = librosa.load(audio_path)
    f = extract_timbral_texture_feature(y, sr, 512 * 43, 512)
    print(audio_path, tuple(f))
