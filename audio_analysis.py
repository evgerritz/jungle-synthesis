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
            freq = fund_freq * partials[i*ncols + j]
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

def timbrel_change(smooth = True, print_out = False):
    # these values obtained by isolating a quiet/loud segment of a solo violin performance
    # running audacity's spectral analysis to get amplitudes for various frequencies
    # and taking the first 30 partials after finding the fundamental frequency
    loud_partial_amps = np.array([-51.93972 , -49.4104  , -46.828625, -56.960018, -64.312508,
       -62.737423, -29.59796 , -62.45232 , -64.916168, -64.447052,
       -70.183189, -72.147018, -65.468178, -70.745598, -59.947681,
       -59.926273, -69.282608, -74.263039, -59.553307, -63.807056,
       -94.537758, -62.298313, -78.946373, -74.343536, -64.548203,
       -82.339119, -84.327194, -70.587006, -74.463646, -83.49617 ])
    quiet_partial_amps = np.array([ -65.825394,  -34.847195,  -52.882778,  -49.364624,  -54.84634 ,
        -51.980816,  -59.094482,  -48.69043 ,  -66.581573,  -54.611156,
        -74.646492,  -59.795284,  -81.939926,  -80.814651,  -77.027634,
        -76.393929,  -89.03508 ,  -72.259262,  -72.2481  ,  -74.552094,
       -106.836113,  -81.866112,  -94.832108,  -79.970268,  -83.187508,
        -89.181671,  -98.206154,  -94.213165,  -98.791794, -100.895943])

    if smooth:
        loud_partial_amps = smooth_data(loud_partial_amps)
        quiet_partial_amps = smooth_data(quiet_partial_amps)

    # plot data
    fig,axs = plt.subplots(3,1)
    axs[0].plot(loud_partial_amps)
    axs[0].set_title("Amplitude vs Partial No for Loud Violin")
    axs[1].plot(quiet_partial_amps)
    axs[1].set_title("Amplitude vs Partial No for Quiet Violin")
    diff = loud_partial_amps - quiet_partial_amps
    axs[2].plot(diff)
    axs[2].set_title("Difference between Loud and Quiet Violin Partials")
    plt.show()

    if print_out:
        # print normalized values for use in supercollider
        print_array((diff + abs(np.min(diff)))/np.max(diff))


if __name__ == '__main__':
    samplerate, samples = wavfile.read('samples/amen_ride.wav')
    if len(samples) == 2: # convert to mono
        samples = samples[:,0]
    
    x_f = np.abs(np.fft.rfft(samples))
    x_freqs = np.fft.rfftfreq(samples.size, 1/samplerate)
    for rel_thresh in np.linspace(0.5, 0.001, 100):
        peaks, _ = signal.find_peaks(x_f, threshold=np.max(x_f)*0.2)
        if len(peaks) > 20 and len(peaks) < 35:
            break

    print('var harmonics = ', [round(f, 2) for f in x_freqs[peaks]], ';', sep='')
    get_envs(x_freqs[peaks], samplerate, samples, smooth=True)
