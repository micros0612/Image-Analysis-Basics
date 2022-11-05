import numpy as np  
import cv2 
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d

def motion_blur(image, degree=10, angle=20):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def add_gaussian_noise(Img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(Img))
	noisy_Img = Img + gauss
	noisy_Img[noisy_Img < 0] = 0
	noisy_Img[noisy_Img > 255] = 255
	return noisy_Img

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h