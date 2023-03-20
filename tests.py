import pygpureg as reg
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time

SIZE = 512
reg.init(image_size=SIZE)

data = tifffile.imread("C:/Users/mgflast/Desktop/220902_srNodes_test.tif")[1:, :SIZE, :SIZE]
n_frames = data.shape[0]
template = data[-1, :, :]


print("Start processing")
timer = time.time_ns()
dxdy = list()
for i in range(n_frames):
    print(i)
    _, shift = reg.phase_correlation(template, data[i])
    dxdy.append(shift)
print(f"Processing speed was {n_frames / (1e-9 * (time.time_ns() - timer))} frames per second.")

dxdy = np.asarray(dxdy)
plt.plot(dxdy[:, 0], label="X shift (px)", color=(0.0, 0.1, 0.7))
plt.plot(dxdy[:, 1], label="Y shift (px)", color=(0.7, 0.1, 0.0))
plt.legend()
plt.show()

