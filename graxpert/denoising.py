import copy
import logging
import time

import numpy as np
import onnxruntime as ort

from graxpert.ai_model_handling import InferenceEngine
from graxpert.application.app_events import AppEvents
from graxpert.application.eventbus import eventbus
from graxpert.ui.ui_events import UiEvents


class MedianNorm:
    def __init__(self, model_threshold):
        self.model_threshold = model_threshold

    def normalize(self, patches):
        # patches = [-1, channels, patch_size, patch_size]
        norm_params = np.empty((patches.shape[0], 2, patches.shape[1], 1, 1), dtype=np.float32)

        _median = np.median(patches, axis=(0, 2, 3), keepdims=True)
        _mad = np.median(np.abs(patches - _median), axis=(0, 2, 3), keepdims=True)

        patches = (patches - _median) / _mad * 0.04
        patches = np.clip(patches, -self.model_threshold, self.model_threshold)

        norm_params[:, 0] = _median
        norm_params[:, 1] = _mad
        
        return patches, norm_params
    
    def denormalize(self, patches, norm_params):
        _median, _mad = norm_params[:, 0], norm_params[:, 1]
        
        patches = patches * _mad / 0.04 + _median
        
        return patches


class ParamsNorm:
    def normalize(self, params):
        return params


normalization_dict = {
    "1.0" : (MedianNorm(model_threshold=1.0), ParamsNorm(), {"patch_size": 256, "stride": 224, "channels": 3, "residuals": False, "channel_last": True}) ,
    "1.1" : (MedianNorm(model_threshold=1.0), ParamsNorm(), {"patch_size": 256, "stride": 224, "channels": 3, "residuals": False, "channel_last": True}) ,
    "2.0" : (MedianNorm(model_threshold=10.0), ParamsNorm(), {"patch_size": 256, "stride": 224, "channels": 3, "residuals": False, "channel_last": True}) ,
    "3.0" : (MedianNorm(model_threshold=10.0), ParamsNorm(), {"patch_size": 256, "stride": 224, "channels": 3, "residuals": False, "channel_last": True}) ,
}

def denoise(image, ai_path, prefs, progress=None):
    logging.info("Starting denoising")

    strenght = prefs.denoise_strength

    global cached_denoised_image
    if cached_denoised_image is not None:
        return (1.0 - strenght) * image + strenght * cached_denoised_image

    num_channels = image.shape[-1]
    if num_channels == 1:
        image = np.repeat(image, 3, axis=-1)
    
    engine = InferenceEngine(
        model_path=ai_path,
        prefs=prefs,
    )

    engine.load_model()
    engine.load_normalization(normalization_dict)

    output = engine.execute(image, None, progress)
    cached_denoised_image = output

    if num_channels == 1:
        output = np.mean(output, axis=-1, keepdims=True)

    logging.info("Finished denoising")

    engine.cleanup()

    return (1.0 - strenght) * image + strenght * output


def reset_cached_denoised_image(event):
    global cached_denoised_image
    cached_denoised_image = None


cached_denoised_image = None
eventbus.add_listener(AppEvents.LOAD_IMAGE_REQUEST, reset_cached_denoised_image)
eventbus.add_listener(AppEvents.CALCULATE_REQUEST, reset_cached_denoised_image)
eventbus.add_listener(UiEvents.APPLY_CROP_REQUEST, reset_cached_denoised_image)
eventbus.add_listener(AppEvents.DENOISE_AI_VERSION_CHANGED, reset_cached_denoised_image)
