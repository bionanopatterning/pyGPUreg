from pyGPUreg import pyGPUreg as reg
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time
from pystackreg import StackReg


sizes = [32, 64, 128, 256, 512, 1024]#, 2048]

times = list()
for SIZE in sizes:
    reg.COS_FILTER_POWER = 1.0
    reg.init(image_size=SIZE)

    data = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/2048.tif")[:, :SIZE, :SIZE]
    n_frames = data.shape[0]
    ## pyGPUreg
    reg.set_template(data[0])
    timer = time.time_ns()
    for i in range(n_frames):
        img, xy = reg.register_to_template(data[i])
    pygpureg_time = n_frames / ((time.time_ns() - timer) * 1e-9)  # fps

    ## stackreg

    sr = StackReg(StackReg.TRANSLATION)
    timer = time.time_ns()
    sr.register_transform_stack(data, reference="first", axis=0)
    stackreg_time = n_frames / ((time.time_ns() - timer) * 1e-9)  # fps

    times.append((pygpureg_time, stackreg_time))

times = np.asarray(times)
plt.plot(sizes, times[:, 0], label="pyGPUreg", color=(0.1, 0.5, 0.1, 1.0), linewidth=1, marker='o', markersize=10)
plt.plot(sizes, times[:, 1], label="StackReg", color=(0.1, 0.1, 0.5, 1.0), linewidth=1, marker='o', markersize=10)
plt.legend()
plt.xlabel("Image size (pixels)")
plt.ylabel("Processing speed (fps)")
plt.ylim([0.0, np.amax(times) * 1.1])
plt.show()

