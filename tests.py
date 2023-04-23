from pyGPUreg import pyGPUreg as reg
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time
from pystackreg import StackReg

SIZE = 256

def templated_registration():
    template = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/img_a.tif")[:SIZE, :SIZE]
    image = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/img_b.tif")[:SIZE, :SIZE]

    reg.set_image_size(SIZE)
    reg.set_template(template)
    timer = time.time_ns()
    registered, shift = reg.register_to_template(image)
    print(f"Registration took: {(time.time_ns() - timer) * 1e-9:.4f} seconds. Shift: {shift}")
    plt.subplot(2, 2, 1)
    plt.imshow(template)
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    plt.subplot(2, 2, 3)
    plt.imshow(registered)
    plt.subplot(2, 2, 4)
    plt.imshow(template - registered)
    plt.show()


def paired_registration():
    template = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/img_a.tif")[:SIZE, :SIZE]
    image = tifffile.imread("C:/Users/mart_/Desktop/pygpureg/img_b.tif")[:SIZE, :SIZE]

    timer = time.time_ns()
    registered, shift = reg.register(template, image)
    print(f"Registration took: {(time.time_ns() - timer) * 1e-9:.4f} seconds. Shift: {shift}")
    plt.subplot(2, 2, 1)
    plt.imshow(template)
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    plt.subplot(2, 2, 3)
    plt.imshow(registered)
    plt.subplot(2, 2, 4)
    plt.imshow(template - registered)
    plt.show()


def pygpureg_vs_stackreg():
    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    times = list()
    for SIZE in sizes:
        reg.COS_FILTER_POWER = 1.0

        data = tifffile.imread("C:/Users/mgflast/Desktop/pygpureg/2048.tif")[:, :SIZE, :SIZE]
        n_frames = 50#data.shape[0]


        timer = time.time_ns()
        for i in range(n_frames):
            img, xy = reg.register(data[0], data[i])
        pygpureg_paired_time = n_frames / ((time.time_ns() - timer) * 1e-9)  # fps

        ## pyGPUreg
        reg.set_template(data[0])
        timer = time.time_ns()
        for i in range(n_frames):
            img, xy = reg.register_to_template(data[i])
        pygpureg_templated_time = n_frames / ((time.time_ns() - timer) * 1e-9)  # fps

        ## stackreg
        sr = StackReg(StackReg.TRANSLATION)
        timer = time.time_ns()
        sr.register_transform_stack(data[:n_frames], reference="first", axis=0)
        stackreg_time = n_frames / ((time.time_ns() - timer) * 1e-9)  # fps

        times.append((pygpureg_templated_time, pygpureg_paired_time, stackreg_time))

    times = np.asarray(times)
    plt.subplot(1,2,1)
    plt.title("Processing speed of pyGPUreg vs. StackReg")
    plt.plot(sizes, times[:, 0], label="pyGPUreg templated", color=(0.1, 0.5, 0.1, 1.0), linewidth=1, marker='o', markersize=5)
    plt.plot(sizes, times[:, 1], label="pyGPUreg paired", color=(0.5, 0.1, 0.1, 1.0), linewidth=1, marker='o', markersize=5)
    plt.plot(sizes, times[:, 2], label="StackReg", color=(0.1, 0.1, 0.5, 1.0), linewidth=1, marker='o', markersize=5)
    plt.legend()
    plt.xlabel("Image size (pixels)")
    plt.ylabel("Processing speed (fps)")
    plt.ylim([-0.1 * np.amax(times), np.amax(times) * 1.1])

    plt.subplot(1,2,2)
    plt.title("Speed gain")
    plt.plot(sizes, 100.0 * times[:, 0] / times[:, 2], label="pyGPUreg templated / StackReg", color=(0.1, 0.5, 0.1, 1.0), linewidth=1, marker='o', markersize=5)
    plt.plot(sizes, 100.0 * times[:, 1] / times[:, 2], label="pyGPUreg paired / StackReg", color=(0.5, 0.1, 0.1, 1.0), linewidth=1, marker='o', markersize=5)
    plt.xlabel("Image size (pixels)")
    plt.ylabel("% of StackReg processing speed")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    reg.init()
    templated_registration()
    paired_registration()