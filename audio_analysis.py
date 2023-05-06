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

def get_xf(samples, samplerate):
    x_f = np.abs(np.fft.rfft(samples))
    x_freqs = np.fft.rfftfreq(samples.size, 1/samplerate)
    return x_f, x_freqs

def get_freq_amps(filename):
    lines = open(filename).readlines()[1:]
    # trim trailing newline
    lines = [line[:-1] for line in lines]
    # split around tab
    lines = [line.split() for line in lines]
    # convert to numbers
    freq_amps = [[float(freq), float(amp)] for freq, amp in lines]
    # round to 3 digits
    freq_amps = [[round(freq,3), round(amp,3)] for freq, amp in freq_amps]
    freq_amps = np.array(freq_amps)
    return freq_amps

def peak_freqs(x_f, x_freqs, min_p = 20, max_p = 35, max_freq=None, dist=1):
    for height_thresh in np.max(x_f)*np.linspace(1, 10, 1000):
        peaks, _ = signal.find_peaks(x_f, height=height_thresh, distance = dist)
        if len(peaks) < 1112:
            print(len(peaks))
        # cut out overlapping ride in bass sample, e.g.
        if max_freq:
            peaks = list(filter(lambda ind: x_freqs[ind] < max_freq, peaks))
        if len(peaks) > min_p and len(peaks) < max_p:
            print('height thresh:', height_thresh)
            break
    else:
        print('failed')
    return x_freqs[peaks]

def print_amp_of_freqs(freq_amps, freqs):
    amps = []
    for partial in freqs:
        closest_partials = list(filter(lambda x: abs(x[0] - partial) < 3, freq_amps))
        if closest_partials:
            closest_partial = closest_partials[-1]
            amps.append(round(closest_partial[1], 3))
        else:
            amps.append(0)

    amps = [amp if amp != 0 else -40.0 for amp in amps]
    print('var amps = ', amps, '.dbamp;', sep='')


def supercollider_arrs(peak_freqs, samples, samplerate, smooth=True):
    print('var harmonics = ', [round(f, 2) for f in peak_freqs], ';', sep='')
    get_envs(peak_freqs, samplerate, samples, smooth)

def read_mono(filename):
    samplerate, samples = wavfile.read(filename)
    if len(samples.shape) == 2: # convert to mono
        samples = samples[:,0]
    return samples, samplerate

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

if __name__ == '__main__':
    samples, samplerate = read_mono('samples/amen_ride.wav')
    freq_amps = get_freq_amps('samples/ride_spectrum.txt')
    #x_f, x_freqs = get_xf(samples, samplerate)
    #print(freq_amps)
    freqs = peak_freqs(freq_amps[:,1], freq_amps[:,0], min_p=10, max_p=20)
    #plt.plot(freq_amps[:,0][:len(freq_amps)//4], freq_amps[:,1][:len(freq_amps)//4])
    #plt.show()
    #plt.plot(freq_amps[:,0], norm(freq_amps[:,1]))
    #plt.show()
    print_amp_of_freqs(freq_amps, freqs)
    supercollider_arrs(freqs, samples, samplerate)


