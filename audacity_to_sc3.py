#!/usr/bin/env python
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        try:
            lines = open(sys.argv[1]).readlines()[1:]
        except:
            print('Error: could not find file:', sys.argv[1])
            sys.exit()

        # trim trailing newline
        lines = [line[:-1] for line in lines]
        # split around tab
        lines = [line.split() for line in lines]
        # convert to numbers
        freq_amps = [[float(freq), float(amp)] for freq, amp in lines]
        # round to 3 digits
        freq_amps = [[round(freq,3), round(amp,3)] for freq, amp in freq_amps]
        freq_amps = np.array(freq_amps)
        if True:
            ind = np.argmin(np.abs(freq_amps[:, 0] - 329.63))
        else:
            ind = np.argmax(freq_amps[:, 1])

        fund_freq = freq_amps[ind, 0]
        print(fund_freq)

        partials = fund_freq * np.array([0.25, 0.5] + list(range(1,20)))
        amps = []
        for partial in partials:
            closest_partials = list(filter(lambda x: abs(x[0] - partial) < 3, freq_amps))
            if closest_partials:
                closest_partial = closest_partials[-1]
                amps.append(round(closest_partial[1], 3))
            else:
                amps.append(0)

        print('var freqs = ', list(partials), ';', sep='')
        print('var amps = ', amps, '.dbamp;', sep='')
    else:
        print('Error: please supply a file name')
