import logging
import cv2 as cv
import numpy as np

from astropy.io import fits

from graxpert.ai_model_handling import InferenceEngine


class LogNorm:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def normalize(self, patches):
        # patches : [-1, patch_size, patch_size, c]
        _min = np.min(patches, axis=(1, 2), keepdims=True)

        patches = np.log(patches - _min + self.epsilon)

        _mean = np.mean(patches, axis=(1, 2), keepdims=True)
        _std = np.std(patches, axis=(1, 2), keepdims=True)

        patches = (patches - _mean) / _std * 0.1

        return patches, (_mean, _std, _min)
    
    def denormalize(self, patches, norm_params):
        # patches : [-1, patch_size, patch_size, c]
        _mean, _std, _min = norm_params
        
        patches = patches * _std / 0.1 + _mean
        
        patches = np.exp(patches) + _min - self.epsilon
        
        return patches


class ParamsNorm:
    def normalize(self, params, model_type=None):
        params[:, 0] = np.clip(params[:, 0], 0.05, 0.95)

        if "stars" in model_type:
            params[:, 1] = np.clip((params[:, 1] / 2.355 - 1.5) / 3.0, 0.05, 0.95)
        else:
            params[:, 1] = np.clip((params[:, 1] / 2.355 - 0.5) / 5.5, 0.05, 0.95)
        
        return params.astype(np.float32)


normalization_dict = {
    "1.0" : (LogNorm(epsilon=1e-5), ParamsNorm(), {"patch_size": 512, "stride": 448, "channels": 1, "residuals": True, "channel_last": False}),
}


def deconvolve(image, ai_path, params, prefs, progress=None):
    logging.info("Starting deconvolution")

    if prefs.deconvolution_apply_luminance_only:
        if image.shape[-1] == 3:
            image_lab = cv.cvtColor(image, cv.COLOR_RGB2Lab)
            image = image_lab[:, :, 0:1]

    engine = InferenceEngine(
        model_path=ai_path,
        prefs=prefs,
    )

    engine.load_model()
    engine.load_normalization(normalization_dict)

    output = engine.execute(image, params, progress)

    if prefs.deconvolution_apply_luminance_only:
        image_lab[:, :, 0] = output
        output = cv.cvtColor(image_lab, cv.COLOR_Lab2RGB)

    engine.cleanup()

    logging.info("Finished deconvolution")

    return output
