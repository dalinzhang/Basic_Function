# function for extracting band power of 1D time-series signal
import numpy as np

def bandpower(data, sf, band, window_sec=None):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2

    Return
    ------
    absolute_bp : float
        Absolute band power.
    relative_bp : float
        Relative power (= absolute band power/total power of the signal).
    psd : array
        Power spectral density Y values (power)
    freqs : array
        Power spectral density X values (frequency)
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    absolute_bp = simps(psd[idx_band], dx=freq_res)
    relative_bp = absolute_bp/simps(psd, dx=freq_res)
    
    return absolute_bp, relative_bp, psd, freqs
