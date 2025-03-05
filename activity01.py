import cv2
import pydicom
import numpy as np
from matplotlib import pyplot as plt

from utils import filepath


def load_dcm(filepath: str):
    """ Load a DICOM file. """
    return pydicom.dcmread(filepath)


def estimate_noisy_pixels(img: np.ndarray):
    """ Estimate the noisy pixels in the background of an image. """
    noise_threshold = 300  # Medido en [T1]
    noise_mask = (img_rest < noise_threshold) * (img_rest > 0)
    return noise_mask


def power_of_signal(signal_or_img: np.ndarray):
    """ Compute the power of a signal. """
    return np.average(np.square(signal_or_img))


def contrast_of_signal(signal_or_img: np.ndarray):
    """ Compute the contrast of a signal. """
    return np.max(signal_or_img) - np.min(signal_or_img)


def compute_snr(signal_power: float, noise_power: float):
    """ Compute the signal-to-noise ratio (SNR) of a signal. """
    return np.sqrt(signal_power) / np.sqrt(noise_power)


def compute_cnr(signal_contrast: float, noise_power: float):
    """ Compute the contrast-to-noise ratio (CNR) of a signal. """
    return signal_contrast / np.sqrt(noise_power)


if __name__ == '__main__':
    filenames = ['PCARDM1_T1_Rest.dcm',
                 'PCARDM1_T1_Stress.dcm']
    dcms = [load_dcm(filepath(n)) for n in filenames]

    dcm_rest = dcms[0]
    img_rest = dcm_rest.pixel_array

    [print(f'{dicom_tag}: {dicom_value}') for dicom_tag, dicom_value in dcm_rest.items()]

    histogram = cv2.calcHist([img_rest.astype('float32')], [0], mask=None, histSize=[256], ranges=[0, 2 ** 16])
    plt.subplot(121), plt.imshow(img_rest, cmap=plt.cm.bone)
    plt.subplot(122), plt.plot(histogram), plt.xticks(np.arange(0, 2 ** 8 + 1, step=2 ** 6),
                                                      np.arange(0, 2 ** 16 + 1, step=2 ** 14), rotation=45)
    plt.show()

    # Cast to float64 to avoid overflow
    img_rest = img_rest.astype('float64')

    # Measure the quality of the image
    noise_mask = estimate_noisy_pixels(img_rest)
    signal_power = power_of_signal(img_rest[~noise_mask])
    noise_power = power_of_signal(img_rest[noise_mask])

    signal_contrast = contrast_of_signal(img_rest[~noise_mask])

    print(f'Signal power: {signal_power} [T1^2].')
    print(f'Signal contrast: {signal_contrast} [T1].')
    print(f'Noise power: {noise_power} [T1^2].')
    print(f'SNR: {compute_snr(signal_power, noise_power)} [1/1].')
    print(f'CNR: {compute_cnr(signal_contrast, noise_power)} [1/1].')

    plt.subplot(221), plt.imshow(~noise_mask)
    plt.subplot(222), plt.imshow(noise_mask)
    plt.subplot(223), plt.imshow(img_rest * (~noise_mask), cmap=plt.cm.bone)
    plt.subplot(224), plt.imshow(img_rest * noise_mask, cmap=plt.cm.bone)
    plt.show()

    # Movement visualization
    movement_thr = 3e3
    movement_img = dcms[0].pixel_array - dcms[1].pixel_array
    movement_mask = np.abs(movement_img) > movement_thr

    plt.subplot(221), plt.imshow(dcms[0].pixel_array, cmap=plt.cm.bone)
    plt.subplot(222), plt.imshow(dcms[1].pixel_array, cmap=plt.cm.bone)
    plt.subplot(223), plt.imshow(movement_img, cmap=plt.cm.bone)
    plt.subplot(224), plt.imshow(movement_mask)
    plt.show()
