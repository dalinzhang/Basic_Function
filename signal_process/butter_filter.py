#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:49:14 2017

@author: dadafly
"""

from scipy.signal import butter, filtfilt

#####################################
# sf is short for sampling frequency
#####################################
def butter_bandpass(lowcut, highcut, sf, pass_type='band', order=3):
	nyq = 0.5 * sf
	low = lowcut / nyq
	high = highcut / nyq
	if 'band' in pass_type:
		b, a = butter(order, [low, high], btype='band', analog=False)
	elif 'high' in pass_type:
		b, a = butter(order, low, btype='highpass', analog=False)
	elif 'low' in pass_type:
		b, a = butter(order, high, btype='lowpass', analog=False)
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, sf, pass_type='band', order=5):
    b, a = butter_bandpass(lowcut, highcut, sf, pass_type, order=order)
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.signal import freqz
	
	# Sample rate and desired cutoff frequencies (in Hz).
	sf = 5000.0
	lowcut = 500.0
	highcut = 1250.0
	pass_type = 'band'
	
	# Plot the frequency response for a few different orders.
	plt.figure(1)
	plt.clf()
	for order in [3, 6, 9]:
	    b, a = butter_bandpass(lowcut, highcut, sf, pass_type, order=order)
	    w, h = freqz(b, a, worN=2000)
	    plt.plot((sf * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
	
	plt.plot([0, 0.5 * sf], [np.sqrt(0.5), np.sqrt(0.5)],
	         '--', label='sqrt(0.5)')
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Gain')
	plt.grid(True)
	plt.legend(loc='best')
	
	# Filter a noisy signal.
	T = 0.05
	nsamples = T * sf
	t = np.linspace(0, T, nsamples, endpoint=False)
	a = 0.02
	f0 = 600.0
	x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
	x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
	x += a * np.cos(2 * np.pi * f0 * t + .11)
	x += 0.03 * np.cos(2 * np.pi * 2000 * t)
	plt.figure(2)
	plt.clf()
	plt.plot(t, x, label='Noisy signal')
	
	y = butter_bandpass_filter(x, lowcut, highcut, sf, pass_type='band', order=6)
	plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
	plt.xlabel('time (seconds)')
	plt.hlines([-a, a], 0, T, linestyles='--')
	plt.grid(True)
	plt.axis('tight')
	plt.legend(loc='upper left')
	
	plt.show()
