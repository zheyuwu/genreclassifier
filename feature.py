#!/usr/bin/env python2

import sys
import timbraltexture
import bpm
import pitch
import librosa

def extract(filename):
    y, sr = librosa.load(filename)
    f = []
    f += timbraltexture.extract_timbral_texture_feature(y, sr, 512 * 43, 512)
    f += bpm.extract_bpm_feature(y)
    f += pitch.extract_pitch_feature(y, sr)
    return tuple(f)

if __name__ == '__main__':
    print(sys.argv[1], extract(sys.argv[1]))
