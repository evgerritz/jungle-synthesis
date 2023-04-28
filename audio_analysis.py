#! /usr/bin/env python
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import scipy.stats as st

def smooth_data(x):
    # smooth to remove vibrato in get_freq, or just remove noise
    # using gaussian kernel with sigma=2
    kern_domain = np.linspace(-2, 2, 21+1)
    gaussian_kernel = st.norm.pdf(kern_domain)
    return np.convolve(x, gaussian_kernel, mode='same')

def get_freq(freq, samplerate, samples, smooth=True):
    """
    returns amplitude values over time for the given frequency
    using a spectrogram/series of STFTs. if smooth is True,
    will use Gaussian smoothing to remove vibrato/noise from 
    the amplitudes
    """
    # good value empirically for maximizing number of freqs and times
    f, t, Pxx = signal.spectrogram(samples,samplerate)
    # find the frequency closest to the target freq at which 
    # fourier samples were taken
    frequency_index = np.argmin(np.abs(f-freq))
    amps = Pxx[frequency_index,:]

    if smooth:
        amps = smooth_data(amps)
    return t, amps

def plot_amp(freq, samplerate, samples, smooth=True):
    plt.ylabel('Power [db]')
    plt.xlabel('Time [sec]')
    plt.title(str(freq) + ' Hz')
    plt.plot(*get_freq(freq, samplerate, samples, smooth))


def plot_amps(fund_freq, samplerate, samples, partials=range(1,31), smooth=True):
    """
    plot an amplitudes over time graph for every partial 
    corresponding to the fundamental frequency
    """
    npartials = len(partials); nrows = npartials//5; ncols = npartials//6
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*8))
    for i in range(nrows):
        for j in range(ncols):
            freq = freqs[i*ncol+j]
            ax = axs[i,j]
            ax.set_ylabel('Power')
            ax.set_xlabel('Time [sec]')
            ax.set_title('Multiple of C#5: ' + str(partials[i*ncols+j]) + ' ' + ' (' + str(freq) + ' Hz' + ')')
            t, amps = get_freq(freq, samplerate, samples, smooth)
            ax.plot(t,amps)

    # adjust spacing
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.7,
                        hspace=1.0)
    plt.show()

def print_array(arr):
    print('[', end = '')
    for x in arr[:-1]:
        print(str(round(x, 3)) + ', ', end='')
    print(round(arr[-1], 3), end='')
    print('], ')

def get_envs(freqs, samplerate, samples, smooth=True):
    """
    prints out a syntactic supercollider array for specifying
    the levels in an envelope
    """
    print('var envs = [', end = '')
    for freq in freqs:
        t, amps = get_freq(freq, samplerate, samples, smooth)
        amps = amps/np.max(amps) # normalize to 0 to 1
        print_array(amps)
    print('];')

def peak_freqs(samples, samplerate, min_p = 20, max_p = 35, max_freq=None):
    x_f = np.abs(np.fft.rfft(samples))
    x_freqs = np.fft.rfftfreq(samples.size, 1/samplerate)
    for rel_thresh in np.linspace(0.5, 0.001, 100):
        peaks, _ = signal.find_peaks(x_f, threshold=np.max(x_f)*rel_thresh)
        # cut out overlapping ride in bass sample, e.g.
        peaks = list(filter(lambda ind: x_freqs[ind] < max_freq, peaks))
        if len(peaks) > min_p and len(peaks) < max_p:
            break
    return x_freqs[peaks]


def read_mono(filename):
    samplerate, samples = wavfile.read(filename)
    if len(samples.shape) == 2: # convert to mono
        samples = samples[:,0]
    return samples, samplerate

def supercollider_arrs(peak_freqs, samples, samplerate, smooth=True):
    print('var harmonics = ', [round(f, 2) for f in peak_freqs], ';', sep='')
    get_envs(peak_freqs, samplerate, samples, smooth)

if __name__ == '__main__':
    bass_samples, samplerate = read_mono('samples/amen_kick_2.wav')
    bass_freqs = peak_freqs(bass_samples, samplerate, min_p=5, max_p=10, max_freq = 500)
    supercollider_arrs(bass_freqs, bass_samples, samplerate)


