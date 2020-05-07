import cv2
import numpy as np


def fft(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # h,w,2
    return dft_shift
def ifft(dft):
    f_idft_shift = np.fft.ifftshift(dft)
    img_back = cv2.idft(f_idft_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back
