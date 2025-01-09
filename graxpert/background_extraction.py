import multiprocessing

multiprocessing.freeze_support()

import logging
from concurrent.futures import wait
from multiprocessing import shared_memory

import cv2 as cv
import numpy as np
import onnxruntime as ort
from astropy.stats import sigma_clipped_stats
from pykrige.ok import OrdinaryKriging
from scipy import interpolate, linalg

from graxpert.ai_model_handling import InferenceEngine
from graxpert.mp_logging import get_logging_queue, worker_configurer
from graxpert.parallel_processing import executor
from graxpert.radialbasisinterpolation import RadialBasisInterpolation


class MedianNorm:
    def __init__(self, padding=8):
        self.padding = padding

    def normalize(self, image):
        norm_params = np.empty([2, 1, 1, 3])

        image = cv.resize(image, dsize=(256 - 2 * self.padding, 256 - 2 * self.padding), interpolation=cv.INTER_AREA)
        image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="reflect")

        _median = np.median(image, axis=(0, 1), keepdims=True)
        _mad = np.median(np.abs(image - _median), axis=(0, 1), keepdims=True)

        image = (image - _median) / _mad * 0.04
        image = np.clip(image, -1.0, 1.0)

        image = image[np.newaxis, ...]

        norm_params[0] = _median
        norm_params[1] = _mad

        return image, norm_params

    def denormalize(self, image, norm_params):
        _median = norm_params[0]
        _mad = norm_params[1]

        image = image[0, ...]

        image = image * _mad / 0.04 + _median

        image = image[self.padding:-self.padding, self.padding:-self.padding]

        return image


class ParamsNorm:
    def normalize(self, params):
        return params


normalization_dict = {
    "1.0" : (MedianNorm(), ParamsNorm(), {"channels": 3, "residuals": False, "channel_last": True}),
}


def gaussian_kernel(sigma=1.0, truncate=4.0):  # follow simulate skimage.filters.gaussian defaults
    ksize = round(sigma * truncate) - 1 if round(sigma * truncate) % 2 == 0 else round(sigma * truncate)
    return (ksize, ksize)


def extract_background(image, ai_path, background_points, downscale_factor, progress=None, prefs=None):
    num_channels = image.shape[-1]

    shm_image = None
    shm_background = None

    if prefs.interpol_type_option == "AI":
        imarray = np.ndarray(image.shape, dtype=np.float32)
        background = np.ndarray(image.shape, dtype=np.float32)
        np.copyto(imarray, image)

        if num_channels == 1:
            imarray = np.repeat(imarray, 3, axis=-1)

        engine = InferenceEngine(
            model_path=ai_path,
            prefs=prefs,
        )

        engine.load_model()
        engine.load_normalization(normalization_dict)

        background = engine.execute_full_image(imarray, None)
        
        engine.cleanup()

        if num_channels == 1:
            background = np.mean(background, axis=-1, keepdims=True)
            imarray = imarray[:, :, 0:1]

        background = cv.GaussianBlur(background, ksize=gaussian_kernel(sigma=20 * prefs.smoothing_option + 3), sigmaX=20 * prefs.smoothing_option + 3)

        background = cv.resize(background, dsize=(imarray.shape[1], imarray.shape[0]), interpolation=cv.INTER_LINEAR)

        if len(background.shape) == 2:
            background = np.expand_dims(background, axis=-1)
        
    else:
        shm_image = shared_memory.SharedMemory(create=True, size=image.nbytes)
        shm_background = shared_memory.SharedMemory(create=True, size=image.nbytes)
        imarray = np.ndarray(image.shape, dtype=np.float32, buffer=shm_image.buf)
        background = np.ndarray(image.shape, dtype=np.float32, buffer=shm_background.buf)
        np.copyto(imarray, image)

        x_sub = np.array(background_points[:, 0], dtype=int)
        y_sub = np.array(background_points[:, 1], dtype=int)

        if progress is not None:
            progress.update(24)

        futures = []
        logging_queue = get_logging_queue()
        for c in range(num_channels):
            futures.insert(
                c,
                executor.submit(
                    interpol,
                    shm_image.name,
                    shm_background.name,
                    c,
                    x_sub,
                    y_sub,
                    image.shape,
                    prefs.interpol_type_option,
                    prefs.smoothing_option,
                    downscale_factor,
                    prefs.sample_size,
                    prefs.RBF_kernel,
                    prefs.spline_order,
                    imarray.dtype,
                    logging_queue,
                    worker_configurer,
                ),
            )
        wait(futures)

        if progress is not None:
            progress.update(48)

    # Correction
    if prefs.corr_type == "Subtraction":
        mean = np.mean(background)
        imarray[:, :, :] = imarray[:, :, :] - background[:, :, :] + mean
    elif prefs.corr_type == "Division":
        for c in range(num_channels):
            mean = np.mean(imarray[:, :, c])
            imarray[:, :, c] = imarray[:, :, c] / background[:, :, c] * mean

    if progress is not None:
        progress.update(8)

    # clip image
    imarray[:, :, :] = imarray.clip(min=0.0, max=1.0)

    image[:] = imarray[:]

    if progress is not None:
        progress.update(8)

    if shm_image is not None:
        shm_image.close()
        shm_image.unlink()
    if shm_background is not None:
        background = np.copy(background)
        shm_background.close()
        shm_background.unlink()

    return background


def calc_mode_dataset(data, x_sub, y_sub, halfsize):

    n = x_sub.shape[0]
    data_padded = np.pad(array=data, pad_width=(halfsize,), mode="reflect")
    subsample = np.zeros(n)

    for i in range(n):
        data_footprint = data_padded[y_sub[i] : y_sub[i] + 2 * halfsize, x_sub[i] : x_sub[i] + 2 * halfsize]
        subsample[i] = sigma_clipped_stats(data=data_footprint, cenfunc="median", stdfunc="std", grow=4)[1]

    return subsample


def interpol(shm_imarray_name, shm_background_name, c, x_sub, y_sub, shape, kind, smoothing, downscale_factor, sample_size, RBF_kernel, spline_order, dtype, logging_queue, logging_configurer):
    logging_configurer(logging_queue)
    logging.info("background_extraction.interpol started")

    try:
        existing_shm_imarray = shared_memory.SharedMemory(name=shm_imarray_name)
        existing_shm_background = shared_memory.SharedMemory(name=shm_background_name)
        imarray = np.ndarray(shape, dtype, buffer=existing_shm_imarray.buf)  # [:,:,channel_idx]
        imarray = imarray[:, :, c]
        background = np.ndarray(shape, dtype, buffer=existing_shm_background.buf)
        shape = imarray.shape

        subsample = calc_mode_dataset(imarray, x_sub, y_sub, sample_size)

        if downscale_factor != 1:
            x_sub = x_sub / shape[1]
            y_sub = y_sub / shape[0]

            shape_scaled = (shape[0] // downscale_factor, shape[1] // downscale_factor)

            x_sub = x_sub * shape_scaled[1]
            y_sub = y_sub * shape_scaled[0]

        else:
            shape_scaled = shape

        if kind == "RBF":
            points_stacked = np.stack([x_sub, y_sub], -1)
            interp = RadialBasisInterpolation(points_stacked, subsample, kernel=RBF_kernel, smooth=smoothing * linalg.norm(subsample) / np.sqrt(len(subsample)))

            # Create background from interpolation
            x_new = np.arange(0, shape_scaled[1], 1)
            y_new = np.arange(0, shape_scaled[0], 1)

            xx, yy = np.meshgrid(x_new, y_new)
            points_new_stacked = np.stack([xx.ravel(), yy.ravel()], -1)

            result = interp(points_new_stacked).reshape(shape_scaled)

        elif kind == "Splines":
            interp = interpolate.bisplrep(y_sub, x_sub, subsample, w=np.ones(len(x_sub)) / np.std(subsample), s=smoothing * len(x_sub), kx=spline_order, ky=spline_order)

            # Create background from interpolation
            x_new = np.arange(0, shape_scaled[1], 1)
            y_new = np.arange(0, shape_scaled[0], 1)
            result = interpolate.bisplev(y_new, x_new, interp)

        elif kind == "Kriging":
            OK = OrdinaryKriging(
                x=x_sub,
                y=y_sub,
                z=subsample,
                variogram_model="spherical",
                verbose=False,
                enable_plotting=False,
            )

            # Create background from interpolation
            x_new = np.arange(0, shape_scaled[1], 1).astype("float64")
            y_new = np.arange(0, shape_scaled[0], 1).astype("float64")

            result = np.zeros(shape_scaled, dtype=np.float32)

            num_it = shape_scaled[0] // 50

            for i in range(num_it):
                result_i, var = OK.execute("grid", xpoints=x_new, ypoints=y_new[i * 50 : (i + 1) * 50], backend="vectorized")
                result[i * 50 : (i + 1) * 50, :] = result_i

            result_i, var = OK.execute("grid", xpoints=x_new, ypoints=y_new[num_it * 50 :], backend="vectorized")
            result[num_it * 50 :, :] = result_i

        else:
            logging.warning("Interpolation method not recognized")
            return

        if downscale_factor != 1:
            result = cv.resize(src=result, dsize=(shape[1], shape[0]), interpolation=cv.INTER_LINEAR)

        background[:, :, c] = result
    except Exception as e:
        logging.exception("Error occured during background_extraction.interpol")

    existing_shm_imarray.close()
    existing_shm_background.close()

    logging.info("background_extraction.interpol finished")
