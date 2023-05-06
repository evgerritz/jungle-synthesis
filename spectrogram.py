import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import glob

files = glob.glob('samples/*.wav')
freqs = []

def fourier_trans(x, samplerate):
    f, t, spec = signal.stft(x, samplerate, nperseg=1000)
    return spec, f, t

for filename in files:
    samplerate, samples = wavfile.read(filename)
    stft = fourier_trans(samples,samplerate)
    plt.pcolormesh(stft[2],stft[1],np.abs(stft[0]), shading='gouraud')
    plt.title(filename)
    plt.savefig(filename + ".png")


