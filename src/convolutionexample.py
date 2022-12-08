# imports
import numpy as np
from imageio import imread, imsave
from scipy import signal
import matplotlib.pyplot as plt

img = imread("../data/millie2.jpg", as_gray=True)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
img_sharp = signal.convolve2d(img, kernel, mode="same")

plt.subplot(121)
plt.title("Original Image", size=25)
plt.imshow(img, cmap="gray")

plt.subplot(122)
plt.title("Image after convolution with a laplacian kernel", size=25)
plt.imshow(img_sharp, cmap="gray", vmin=10, vmax=14)

plt.show()
