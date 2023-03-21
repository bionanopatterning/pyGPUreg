from pyGPUreg import pyGPUreg as reg
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time

SIZE = 256
reg.COS_FILTER_POWER = 1.0
reg.init(image_size=SIZE)

template = tifffile.imread("C:/Users/mart_/Desktop/220902_srNodes_test.tif", key=1)[:SIZE, :SIZE]

reg.set_template(template)
for i in range(10):
    img, xy = reg.register_to_template(tifffile.imread("C:/Users/mart_/Desktop/pygpureg/2048.tif", key=i+1)[:SIZE, :SIZE])
    print(xy)



