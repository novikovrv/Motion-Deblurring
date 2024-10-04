import argparse
import cv2
import numpy as np
from numpy.fft import fft2, ifft2



def wiener_filter(img, kernel, K):
  kernel /= np.sum(kernel)
  dummy = np.copy(img)
  dummy = fft2(dummy)
  kernel = fft2(kernel, s = img.shape)
  kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
  dummy = dummy * kernel
  dummy = np.abs(ifft2(dummy))
  return dummy



def apply_kernel(img, kernel, noise=10):
    noise = 10 ** (-0.1 * noise)
    kernel /= kernel.sum()

    kernel_pad = np.zeros_like(img[:, :, 0])
    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel

    PSF = cv2.dft(kernel_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
    PSF2 = (PSF ** 2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
    RES = cv2.mulSpectrums(img, iPSF, 0)

    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    res = np.roll(res, -kh // 2, 0)
    res = np.roll(res, -kw // 2, 1)

    return res


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True)
parser.add_argument('--output_path', default='result.png')
parser.add_argument('--kernel_path')
if __name__ == '__main__':
    args = parser.parse_args()

    img = cv2.imread(args.image_path, 0)
    img = np.float32(img) / 255.0
    img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2GRAY)
    kernel = cv2.imread("kernel.png", cv2.IMREAD_GRAYSCALE)
    kernel = kernel.astype(np.float32)

    deconvolved_image = wiener_filter(image, kernel, 0.01)
    cv2.imwrite("res1.png", deconvolved_image)

    res = apply_kernel(img, kernel, noise=10)
    cv2.imwrite("res2.png", res * 255)
