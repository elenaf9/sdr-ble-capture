import numpy as np
import matplotlib.pyplot as plt

sample_rate = 40e6
center_freq = 2420e6
fft_size = 8192

c = np.fromfile("../../data/capture.dat", dtype=np.float64)
num_rows = len(c) // fft_size
spectrogram = c.reshape((num_rows, fft_size))

print(np.max(spectrogram))
print(np.min(spectrogram))

extent = [(center_freq + sample_rate/-2)/1e6, (center_freq + sample_rate/2)/1e6, len(c)/sample_rate, 0]

plt.imshow(spectrogram, aspect='auto', extent=extent)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")

cbar = plt.colorbar()
cbar.set_label("Relative Power [dB]")

# plt.show()
plt.savefig("data/capture.png", dpi=1000)
plt.close()


