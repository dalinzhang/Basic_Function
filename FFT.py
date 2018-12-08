#! /usr/bin/python3
import numpy as np

def FFT(x, freq):
	freq_spectrum = np.fft.fft(x)
	freq_power = np.abs(freq_spectrum)**2
	freqs = np.fft.fftfreq(x.size, 1/freq)
	return freq_power, freqs


def band_power_extract(x, freq, low, high):
	freq_power, freqs = FFT(x, freq)
	idx = (freqs >= low) & (freqs <= high)
	band_power_spectrum = freq_power[idx]
	band_power = sum(band_power_spectrum)
	return band_power
