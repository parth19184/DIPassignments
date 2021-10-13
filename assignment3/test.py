import cv2
import matplotlib.pyplot as plt
import numpy as np
import cmath

def DFT2D(image):
    data = np.asarray(image)
    print(image.shape)
    M, N = image.shape # (img x, img y)
    dft2d = np.zeros((M,N))
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(- 2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_matrix +=  data[m,n] * e
            dft2d[k,l] = sum_matrix
    return dft2d

camera_man = cv2.imread('input_image.jpg', 0)
plt.imshow(camera_man)
img = camera_man.resize((50,50))
dft_img = DFT2D(camera_man)
plt.imshow(dft_img.real)
plt.show()