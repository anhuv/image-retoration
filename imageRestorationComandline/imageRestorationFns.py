import numpy as np

def full_inverse_filter (image, psf):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]

    psf_padded = np.zeros_like (image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum (psf_padded)

    result = np.zeros_like (image)

    psf_dft = dft_2d (psf_padded)

    psf_dft[psf_dft == 0] = 0.00001

    for i in range (0, 3):
        image_dft = dft_2d (image[:, :, i]);
        temp = np.abs (idft_2d (np.divide (image_dft, psf_dft)))
        result[:, :, i] = temp.astype (np.uint8)

    return result

def truncated_inverse_filter (image, psf, R):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]

    psf_padded = np.zeros_like (image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum (psf_padded)

    result = np.zeros_like (image)
    psf_dft = dft_2d (psf_padded)

    psf_dft[psf_dft == 0] = 0.00001

    lpf = get_butterworth_lpf (image.shape[0], image.shape[1], 10, R)
    lpf = shift_dft (lpf)

    for i in range (0, 3):
        image_dft = dft_2d (image[:, :, i])
        temp = np.abs (idft_2d (np.multiply (np.divide (image_dft, psf_dft), lpf)))
        result[:, :, i] = temp.astype (np.uint8)

    return result

def weiner_filter (image, psf, K):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]

    psf_padded = np.zeros_like (image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum (psf_padded)

    result = np.zeros_like (image)
    psf_dft = dft_2d (psf_padded)

    for i in range (0, 3):
        image_dft = dft_2d (image[:, :, i])
        psf_dft_abs = np.square (np.abs (psf_dft))
        temp1 = np.divide (psf_dft_abs, psf_dft_abs + K * np.ones_like (image_dft))
        temp2 = np.divide (image_dft, psf_dft)
        temp = np.abs (idft_2d (np.multiply (temp1, temp2)))
        result[:, :, i] = temp.astype (np.uint8)

    return result;

def constrained_ls_filter (image, psf, gamma):
    psf_M = psf.shape[0]
    psf_N = psf.shape[1]

    psf_padded = np.zeros_like (image[:, :, 0])
    psf_padded[0:psf_M, 0:psf_N] = psf
    psf_padded = psf_padded / np.sum (psf_padded)

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
    laplacian_padded = np.zeros_like (image[:, :, 0], dtype=int)
    laplacian_padded[0:3, 0:3] = laplacian

    result = np.zeros_like (image)

    psf_dft = dft_2d (psf_padded)

    laplacian_dft = dft_2d (laplacian_padded)

    laplacian_dft_abs = np.square (np.abs (laplacian_dft))
    psf_dft_abs = np.square (np.abs (psf_dft))
    temp1 = np.divide (psf_dft_abs, psf_dft_abs + gamma * laplacian_dft_abs)

    for i in range (0, 3):
        image_dft = dft_2d (image[:, :, i])
        temp2 = np.divide (image_dft, psf_dft)
        temp = np.abs (idft_2d (np.multiply (temp1, temp2)))
        result[:, :, i] = temp.astype (np.unit8)

    return result

def psnr (image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]

    mse = np.sum (np.square (image_original - image_restored)) / (M * N) + 0.000001
    max = 255**2
    psnr = 10 * np.log10 (max/mse)

    return psnr

def ssim (image_original, image_restored):
    M = image_original.shape[0]
    N = image_original.shape[1]
    one = np.ones_like (image_original)

    mean_original = np.sum (image_original) / (M * N)
    mean_restored = np.sum (image_original) / (M * N)

    var_original = np.sum (np.square (image_original - mean_original * one)) / (M * N)
    var_restored = np.sum (np.square (image_restored - mean_restored * one)) / (M * N)

    cross_correlation = np.sum (np.multiply (image_original - mean_original * one, image_restored - mean_restored * one)) / (M * N)

    sd_original = np.sqrt (var_original)
    sd_restored = np.sqrt (var_restored)

    C1 = 1
    C2 = 0.01
    C3 = 0.01

    l = (2 * mean_original * mean_restored + C1) / (mean_original**2 + mean_restored**2 + C1)
    c = (2 * sd_original * sd_restored + C2) / (var_original + var_restored + C2)
    s = (cross_correlation + C3) / (sd_original * sd_restored + C3)

    ssim = l*c*s

    return ssim

def get_butterworth_lpf (M, N, oder, radius):
    m = range (0, M)
    m0 = int(M / 2) * np.ones (M)
    n = range (0, N)
    n0 = int(N / 2) * np.ones (N)

    r2 = radius ** 2

    row = np.tile ((np.power (m - m0, 2*np.ones(M)) / r2).reshape (M, 1), (1, N))
    column = np.tile ((np.power (n-n0, 2*np.ones(N)) / r2).reshape (1, N), (M, 1))

    butterworth_lpf = np.divide (np.ones_like (row), np.power (row+column, oder*np.ones_like (row)) + np.ones_like (row))

    return butterworth_lpf

def shift_dft (image):
    shifted_dft = np.fft.fftshift (image)
    return shifted_dft

def dft_2d (image):
    dft = np.fft.fft2 (image)
    return dft

def idft_2d (dft):
    idft = np.fft.ifft2 (dft)
    return idft

