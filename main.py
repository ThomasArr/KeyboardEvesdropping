# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api

from AudioUtil import AudioUtil


def test(filename):
    fs, data = wavfile.read(filename)  # load the data
    a = data.T[0]  # this is a two channel soundtrack, I get the first track
    b = [(ele / 2 ** 8.) * 2 - 1 for ele in a]  # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b)  # calculate fourier transform (complex numbers list)
    d = int(len(c) / 2)  # you only need half of the fft list (real signal symmetry)
    plt.subplot(2)
    plt.plot(abs(c[:(d - 1)]), 'r')
    plt.plot(abs(a[:(d - 1)]), 'b')
    plt.show()


audioUtil = AudioUtil()

file = audioUtil.open("./data/z1.wav")

file = audioUtil.pad_trunc(file, 400, 6600)

sig, sr = file
plt.plot(sig[0])
plt.show()