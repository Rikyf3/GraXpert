import copy
import logging
import numpy as np
import onnxruntime as ort

from graxpert.ai_model_handling import InferenceEngine
from graxpert.application.app_events import AppEvents
from graxpert.application.eventbus import eventbus


class LogNorm:
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon

    def normalize(self, patches):
        # patches : [num, channels, patch_size, patch_size]
        norm_params = np.empty((patches.shape[0], 3, patches.shape[1], 1, 1), dtype=np.float32)

        _min = np.min(patches, axis=(2, 3), keepdims=True)

        patches = np.log(patches - _min + self.epsilon)

        _mean = np.mean(patches, axis=(2, 3), keepdims=True)
        _std = np.std(patches, axis=(2, 3), keepdims=True)

        patches = (patches - _mean) / _std * 0.1

        norm_params[:, 0] = _mean
        norm_params[:, 1] = _std
        norm_params[:, 2] = _min

        return patches, norm_params
    
    def denormalize(self, patches, norm_params):
        _mean, _std, _min = norm_params[:, 0], norm_params[:, 1], norm_params[:, 2]
        
        patches = patches * _std / 0.1 + _mean
        
        patches = np.exp(patches) + _min - self.epsilon
        
        return patches


class ParamsNorm:
    def normalize(self, params, model_type=None):
        params[:, 0] = params[:, 0] * 0.95
        
        if "stars" in model_type:
            params[:, 1] = np.clip((params[:, 1] / 2.355 - 1.5) / 3.0, 0.05, 0.95)
        else:
            params[:, 1] = np.clip((params[:, 1] / 2.355 - 0.5) / 5.5, 0.05, 0.95)
        
        return params.astype(np.float32)


normalization_dict = {
    "1.0" : (LogNorm(epsilon=1e-2), ParamsNorm(), {"patch_size": 512, "stride": 448, "channels": 1, "residuals": True, "channel_last": False}),
}


# TODO : Add events star and end
def deconvolve(image, ai_path, params, prefs, progress=None):
    logging.info("Starting deconvolution")

    engine = InferenceEngine(
        model_path=ai_path,
        prefs=prefs,
    )

    engine.load_model()
    engine.load_normalization(normalization_dict)

    output = engine.execute(image, params, progress)

    engine.cleanup()

    logging.info("Finished deconvolution")

    return output
