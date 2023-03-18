## pyGPUreg ##
pyGPUreg is a minimal Python module for GPU-accelerated image registration designed for use in drift correction of microscopy images. On a PC with an Nvidia Quadro P1000, pyGPUreg can process around 50 images (512 x 512 pixels) per second. That processing entails both detecting the image shift and correcting it.

The module works by computing the phase correlation of input images on the GPU. Our implementation of FFTs on the GPU is largely based on the discussion and code in Fynn-Jorin Flügge's 'Realtime GPGPU FFT Ocean Water Simulation' (doi.org/10.15480/882.1436). pyGPUfit uses OpenGL, so a CUDA compatible card is not required.  

### Usage ###
For now, pyGPUreg only works on images that are square and with size 2^n (e.g. 128, 256, 512, etc. pixel width and height)

#### Initialization ####
Prior to calling any of the core functions, pyGPUreg has to be initialized.

```
import pyGPUreg as reg

reg.init(create_window=True)
```

We use glfw to create an OpenGL context, and since glfw requires a window to be opened we create a hidden window in the init function. You can ignore this. To use pyGPUreg in an application that already has an OpenGL context, call init() with create_window=False.

#### Drift correction ####
```
import pygpureg as reg
import tifffile
import matplotlib.pyplot as plt
import time

reg.init(image_size=256)

template = tifffile.imread("template.tif")
image = tifffile.imread("subject.tif")

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
```
![](res/Registration_example.PNG)
```
Registration took: 0.0090 seconds.
```

#### 2D FFT on the GPU ####
Below is an example of using pyGPUreg to compute a 2D FFT of a grayscale image 
```
import pygpureg as reg
import tifffile
import numpy as np
import matplotlib.pyplot as plt

reg.init()
img = tifffile.imread("pom.tif")
img = img - np.mean(img)

plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("Input image")

plt.subplot(1,3,2)
fft = np.fft.fftshift(reg.gpu_fft(img), axes=(0, 1))
plt.imshow(np.absolute(fft))
plt.title("pyPGUfit FFT")

plt.subplot(1,3,3)
fft = np.fft.fftshift(np.fft.fft2(img), axes=(0, 1))
plt.imshow(np.absolute(fft))
plt.title("numpy FFT")
plt.show()
```

![](res/FFT_example.PNG)
