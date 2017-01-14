import sys
import pywt
import librosa
import numpy as np
import collections
import multiprocessing
from utils import *


WINDOW_SIZE = 65536
SR = 22050


def parallel(window):
    db4 = pywt.Wavelet('db4')
    coffs = pywt.wavedec(window, db4, level=10)
    bands = list()
    for lv in range(1, len(coffs)):
        bands.append(pywt.upcoef('d', coffs[-lv], db4, level=lv))
    bands.append(pywt.upcoef('a', coffs[0], db4, level=len(coffs) - 1))

    scale = 4
    xs = np.zeros(len(window) / scale)
    for band in bands:
        band = np.abs(band[:len(window)])
        ys = np.zeros(len(band))
        ys[0] = band[0]
        for i, x in enumerate(band[1:]):
            ys[i] = (1 - 0.99) * x + 0.99 * ys[-1]
        ds = np.zeros(len(xs))
        for k in range(len(ds)):
            ds[k] = ys[k * scale]
        ds -= np.mean(ds)
        xs += ds

    ys = np.zeros(len(xs))
    for k in range(1, len(xs)):
        xsp = np.zeros(len(xs))
        xsp[:-k] = xs[k:]
        ys[k] = np.dot(xs, xsp) / len(xs)

    tmp2 = librosa.util.peak_pick(ys, 1024 / scale, 1024 / scale, 2048 / scale, 2048 / scale, 0, 1024 / scale)
    if len(tmp2) == 0:
        return []

    top3 = []
    for signal, bpm in reversed(sorted(zip(ys[tmp2], 60.0 / (tmp2 * (float(scale) / SR))))):
        if bpm >= 40 and bpm <= 200:
            top3.append((round(bpm / 2) * 2, signal))
            if len(top3) == 3:
                break

    return top3


def calc(data):
    hist = np.zeros(300)
    pool = multiprocessing.Pool(16)
    tops = pool.map(parallel, SliceWindow(data, WINDOW_SIZE, WINDOW_SIZE / 2))
    for top3 in tops:
        for bpm, signal in top3:
            hist[int(bpm)] += signal

    snd, fst = sorted(librosa.util.peak_pick(hist, 5, 5, 10, 10, 0, 5), key=lambda x: hist[x])[-2:]
    a0 = hist[fst] / (hist[fst] + hist[snd])
    a1 = hist[snd] / (hist[fst] + hist[snd])
    ra = hist[snd] / hist[fst]
    p1 = 1.0 / fst
    p2 = 1.0 / snd
    s = np.sum(hist)
    return (a0, a1, ra, p1, p2, s)
    

if __name__ == '__main__':
    data, sr = librosa.load(sys.argv[1])
    assert sr == SR
    print(sys.argv[1], calc(data))

    #onset_env = librosa.onset.onset_strength(data, sr=sr)
    #tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)
    #print(tempo)
