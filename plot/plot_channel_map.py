import numpy as np
import matplotlib.pyplot as plt

channels = 20
window_size = 0.1

c = np.fromfile("../../data/channel_capture.dat", dtype=np.float32)
num_rows = len(c) // (3*channels)
channel_avg = c[0::3].reshape((num_rows, channels))
channel_min = c[1::3].reshape((num_rows, channels))
channel_max = c[2::3].reshape((num_rows, channels))

mins = np.nanmin(channel_avg, axis=0)
maxs = np.nanmax(channel_avg, axis=0)
avgs = np.nanmean(channel_avg, axis=0)

# Relative to noise floor
channel_avg_relative = (channel_avg - mins)
# Scale to [0,1]
# channel_avg_relative = (channel_avg - mins) / (maxs - mins)

extent = [0, channels, num_rows/(1/window_size), 0]
for (label, data) in [("average", channel_avg_relative), ("min", channel_min), ("max", channel_max)]:
    plt.imshow(data, aspect='auto', extent=extent)
    plt.xlabel("Channel")
    plt.ylabel("Time [s]")

    cbar = plt.colorbar()
    cbar.set_label(f"Relative Power {label}[dB]")

    # plt.show()
    plt.savefig(f"data/ch_{label}.png", dpi=1000)
    plt.close()


