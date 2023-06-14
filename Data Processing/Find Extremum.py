from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# load heartbeat signal
signal = np.load('data_1.npy')

# find the peaks and valleys in the signal
peaks, _ = find_peaks(signal, height=0)
valleys, _ = find_peaks(-signal, height=0)

# plot the signal, peaks, and valleys
plt.plot(signal)
plt.plot(peaks, signal[peaks], "x", label="peaks")
plt.plot(valleys, signal[valleys], "o", label="valleys")
plt.legend()
plt.show()
