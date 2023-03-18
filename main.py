import pygpufit
import tifffile
import time

# TODO: negative shifts; e.g. when size is 128x128 pixels and shift is 124, it should be -4.

SIZE = 128

img = tifffile.imread("C:/Users/mgflast/Desktop/img_a.tif")[:SIZE, :SIZE]
img2 = tifffile.imread("C:/Users/mgflast/Desktop/img_b.tif")[:SIZE, :SIZE]



pygpufit.init(image_size=SIZE)
start_time = time.time()
regd = pygpufit.cross_correlation(img2, img)
print(f"Cost: {time.time() - start_time} seconds")
pygpufit.terminate()

# import matplotlib.pyplot as plt
#
# plt.subplot(1,3,1)
# plt.imshow(img)
# plt.subplot(1,3,2)
# plt.imshow(img2)
# plt.subplot(1,3,3)
# plt.imshow(regd)
# plt.show()
