import sys
import librosa
from utils import *


def calc(data):
    assert len(data) >= 65536

    for window in SliceWindow(data, 65536):
        pass


if __name__ == '__main__':
    data, sr = librosa.load(sys.argv[1])
    assert sr == 22050
    calc(data)
