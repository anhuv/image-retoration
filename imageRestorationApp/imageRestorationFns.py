# ---------------------------------------------------------------#
# __name__ = "ImageRestoration"
# __author__ = "Chung Duc Nguyen Dang"
# __version__ = "1.0"
# __email__ = "duc.ndc172484@sis.hust.edu.vn"
# __status__ = "Development"
# ---------------------------------------------------------------#

# All array operations are performed using numpy library
import numpy as np


# Implementation of all image processing functions

# Code for computing full inverse - own code
def full_inverse_filter(image, psf):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)

    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # replace 0 value if present in H, to avoid division by zero
    psf_dft[psf_dft == 0] = 0.00001

    # compute F = G/H for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp = np.abs(idft_2d(np.divide(image_dft, psf_dft)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# Code for computing truncated inverse - own code
def truncated_inverse_filter(image, psf, R):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)
    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # replace 0 value if present in H, to avoid division by zero
    psf_dft[psf_dft == 0] = 0.00001

    # compute frequency domain Butterworth LPF of order 10 - L
    lpf = get_butterworth_lpf(image.shape[0], image.shape[1], 10, R)
    lpf = shift_dft(lpf)

    # compute F = (G/H)*L for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp = np.abs(idft_2d(np.multiply(np.divide(image_dft, psf_dft), lpf)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# Code for computing weiner filter - own code
def weiner_filter(image, psf, K):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    result = np.zeros_like(image)
    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)

    # compute F = (G/H) * (|H|^2 / (|H|^2 + K)) for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        psf_dft_abs = np.square(np.abs(psf_dft))
        temp1 = np.divide(psf_dft_abs, psf_dft_abs + K * np.ones_like(image_dft))
        temp2 = np.divide(image_dft, psf_dft)
        temp = np.abs(idft_2d(np.multiply(temp1, temp2)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# Code for computing constrained l s filter - own code
def constrained_ls_filter(image, psf, gamma):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]
    # zero pad the psf in space domain to match the image size
    psf_padded = np.zeros_like(image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum(psf_padded)

    # define laplacian matrix and zero pad it
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
    laplacian_padded = np.zeros_like(image[:, :, 0], dtype=int)
    laplacian_padded[0:3, 0:3] = laplacian

    result = np.zeros_like(image)
    # compute dft of psf - H
    psf_dft = dft_2d(psf_padded)
    # compute dft of laplacian - P
    laplacian_dft = dft_2d(laplacian_padded)

    laplacian_dft_abs = np.square(np.abs(laplacian_dft))
    psf_dft_abs = np.square(np.abs(psf_dft))
    temp1 = np.divide(psf_dft_abs, psf_dft_abs + gamma * laplacian_dft_abs)

    # compute F = (G/H) * (|H|^2 / (|H|^2 + gamma * P)) for each channel i.e. R, G and B separately
    for i in range(0, 3):
        image_dft = dft_2d(image[:, :, i])
        temp2 = np.divide(image_dft, psf_dft)
        temp = np.abs(idft_2d(np.multiply(temp1, temp2)))
        result[:, :, i] = temp.astype(np.uint8)

    return result


# Code for computing psnr - own code
def psnr(image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]

    # compute psnr as 10log10(MAX^2/MSE)
    mse = np.sum(np.square(image_original - image_restored)) / (M * N) + 0.0000001
    max = 255**2
    psnr = 10 * np.log10(max/mse)

    return psnr


# Code for computing ssim - own code
def ssim(image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]
    one = np.ones_like(image_original)

    # compute mean of original and restored image
    mean_original = np.sum(image_original) / (M * N)
    mean_restored = np.sum(image_restored) / (M * N)

    # compute variance of original and restored image
    var_original = np.sum(np.square(image_original - mean_original * one)) / (M * N)
    var_restored = np.sum(np.square(image_restored - mean_restored * one)) / (M * N)

    # compute cross correlation between original and restored image
    cross_correlation = np.sum(np.multiply(image_original - mean_original * one,
                                           image_restored - mean_restored * one)) / (M * N)

    # compute standard deviation of original and restored image
    sd_original = np.sqrt(var_original)
    sd_restored = np.sqrt(var_restored)

    # define constants c1, c2, c3
    C1 = 1
    C2 = 0.01
    C3 = 0.01

    # compute l, c, s
    l = (2 * mean_original * mean_restored + C1) / (mean_original**2 + mean_restored**2 + C1)
    c = (2 * sd_original * sd_restored + C2) / (var_original + var_restored + C2)
    s = (cross_correlation + C3) / (sd_original * sd_restored + C3)

    # compute ssim = lcs
    ssim = l*c*s

    return ssim


# Code for generating butter worth filter frequency response - own code
def get_butterworth_lpf(M, N, order, radius):
    m = range(0, M)
    m0 = int(M / 2) * np.ones(M)
    n = range(0, N)
    n0 = int(N / 2) * np.ones(N)

    r2 = radius ** 2

    # compute butterworth lpf frequency domain representation as 1 / (1 + (x-x0)^2 + (y-y0)^2 / D0^2)^n)
    row = np.tile((np.power(m - m0, 2 * np.ones(M)) / r2).reshape(M, 1), (1, N))
    column = np.tile((np.power(n - n0, 2 * np.ones(N)) / r2).reshape(1, N), (M, 1))

    butterworth_lpf = np.divide(np.ones_like(row),
                                np.power(row + column, order * np.ones_like(row)) + np.ones_like(row))

    return butterworth_lpf


def shift_dft(image):
    shifted_dft = np.fft.fftshift(image)
    return shifted_dft


def dft_2d(image):
    dft = np.fft.fft2(image)
    return dft


def idft_2d(dft):
    idft = np.fft.ifft2(dft)
    return idft


# ---------------- Own Implementation of 2D DFT ----------------
# def dft_2d(image):
#     # Capture no of rows and columns in image
#     M = image.shape[0]
#     N = image.shape[1]
#
#     x = np.array(range(0, M))
#     y = np.array(range(0, N))
#
#     matM = M * np.ones(M)
#     matN = N * np.ones(N)
#
#     expM = np.exp((-1j * 2 * np.pi / M) * x)
#     expN = np.exp((-1j * 2 * np.pi / N) * y)
#
#     dft = np.zeros_like(image, dtype=complex)
#     dft_temp = np.zeros_like(image, dtype=complex)
#
#     for v in range(0, N):
#         wN = expN[np.mod(v * y, matN).astype(int)].reshape(1, N)
#         dft_temp[:, v] = np.sum(np.multiply(image, np.tile(wN, (M, 1))), axis=1)
#
#     for u in range(0, M):
#         wM = expM[np.mod(u * x, matM).astype(int)].reshape(M, 1)
#         dft[u, :] = np.sum(np.multiply(dft_temp, np.tile(wM, (1, N))), axis=0)
#
#     return dft
#
# # ---------------- Own Implementation of 2D IDFT ----------------
# def idft_2d(dft):
#     # Capture no of rows and columns in image
#     M = dft.shape[0]
#     N = dft.shape[1]
#
#     x = np.array(range(0, M))
#     y = np.array(range(0, N))
#
#     matM = M * np.ones(M)
#     matN = N * np.ones(N)
#
#     expM = np.exp((1j * 2 * np.pi / M) * x)
#     expN = np.exp((1j * 2 * np.pi / N) * y)
#
#     idft = np.zeros_like(dft, dtype=complex)
#     idft_temp = np.zeros_like(dft, dtype=complex)
#
#     for v in range(0, N):
#         wN = expN[np.mod(v * y, matN).astype(int)].reshape(1, N)
#         idft_temp[:, v] = np.sum(np.multiply(dft, np.tile(wN, (M, 1))), axis=1)
#
#     for u in range(0, M):
#         wM = expM[np.mod(u * x, matM).astype(int)].reshape(M, 1)
#         idft[u, :] = np.sum(np.multiply(idft_temp, np.tile(wM, (1, N))), axis=0) / (M*N)
#
#     return idft
