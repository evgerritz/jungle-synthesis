import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import glob

files = glob.glob('samples/*.wav')
for file
    samplerate, samples = wavfile.read(filename)
