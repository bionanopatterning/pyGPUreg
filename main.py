import pygpureg
import time
import numpy as np
import matplotlib.pyplot as plt


# template = tifffile.imread("C:/Users/mart_/Desktop/a.tif").astype(float)
# data = tifffile.imread("C:/Users/mart_/Desktop/a.tif").astype(float)
#
# pygpureg.init()
# start_time = time.time()
# regd, _ = pygpureg.cross_correlation(template, data, interpolation_mode=pygpureg.INTERPOLATION_MODE_LINEAR, edge_mode=pygpureg.EDGE_MODE_REFLECT)
# print(f"Cost: {time.time() - start_time} seconds")

import pygpureg as reg
import tifffile
import matplotlib.pyplot as plt
import time

reg.init()

template = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/template.tif")
image = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/subject.tif")

timer = time.time()
registered_image, shift = reg.phase_correlation(template, image, apply_shift=True, subpixel_mode=pygpureg.SUBPIXEL_MODE_COM, edge_mode=pygpureg.EDGE_MODE_ZERO, interpolation_mode=pygpureg.INTERPOLATION_MODE_LINEAR)
print(f"Registration took: {time.time() - timer:.4f} seconds.")

plt.subplot(2,2,1)
plt.imshow(template)
plt.title("Template")
plt.subplot(2,2,2)
plt.imshow(image)
plt.title("Image to be registered")
plt.subplot(2,2,3)
plt.imshow(registered_image)
plt.title("Registered image")
plt.subplot(2,2,4)
plt.imshow(template - registered_image)
plt.title("Template minus registered image")
plt.show()


