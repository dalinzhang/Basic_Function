import numpy as np


def windows(data, size, step):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += step


def segment_signal_without_transition(data, window_size, step):
'''
data has shape of [n_timelength]
'''
    segments = []
    for (start, end) in windows(data, window_size, step):
        if(len(data[start:end]) == window_size):
            segments = segments + [data[start:end]]
    return np.array(segments)


def segment_dataset(X, window_size, step):
'''
X has shape of [n_sample, n_timelength]
'''
    win_x = []
    for i in range(X.shape[0]):
        win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
    win_x = np.array(win_x)
    return win_x

