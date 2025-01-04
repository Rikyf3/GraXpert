import logging
import os
import re
import shutil
import zipfile

import onnxruntime as ort
from appdirs import user_data_dir
from minio import Minio
from packaging import version

import numpy as np
import time
from graxpert.application.app_events import AppEvents
from graxpert.application.eventbus import eventbus

try:
    from graxpert.s3_secrets import endpoint, ro_access_key, ro_secret_key

    client = Minio(endpoint, ro_access_key, ro_secret_key)
except Exception as e:
    logging.exception(e)
    client = None

from graxpert.ui.loadingframe import DynamicProgressThread

ai_models_dir = os.path.join(user_data_dir(appname="GraXpert"), "ai-models")
bge_ai_models_dir = os.path.join(user_data_dir(appname="GraXpert"), "bge-ai-models")

# old ai-models folder exists, rename to 'bge-ai-models'
if os.path.exists(ai_models_dir):
    logging.warning(f"Older 'ai_models_dir' {ai_models_dir} exists. Renaming to {bge_ai_models_dir} due to introduction of new denoising models in GraXpert 3.")
    try:
        os.rename(ai_models_dir, bge_ai_models_dir)
    except Exception as e:
        logging.error(f"Renaming {ai_models_dir} to {bge_ai_models_dir} failed. {bge_ai_models_dir} will be newly created. Consider deleting obsolete {ai_models_dir} manually.")

os.makedirs(bge_ai_models_dir, exist_ok=True)

deconvolution_object_ai_models_dir = os.path.join(user_data_dir(appname="GraXpert"), "deconvolution-object-ai-models")
os.makedirs(deconvolution_object_ai_models_dir, exist_ok=True)
deconvolution_stars_ai_models_dir = os.path.join(user_data_dir(appname="GraXpert"), "deconvolution-stars-ai-models")
os.makedirs(deconvolution_stars_ai_models_dir, exist_ok=True)
denoise_ai_models_dir = os.path.join(user_data_dir(appname="GraXpert"), "denoise-ai-models")
os.makedirs(denoise_ai_models_dir, exist_ok=True)


# ui operations
def list_remote_versions(bucket_name):
    if client is None:
        return []
    try:
        objects = client.list_objects(bucket_name)
        versions = []

        for o in objects:
            tags = client.get_object_tags(o.bucket_name, o.object_name)
            if tags is not None and "ai-version" in tags:
                versions.append(
                    {
                        "bucket": o.bucket_name,
                        "object": o.object_name,
                        "version": tags["ai-version"],
                    }
                )
        return versions

    except Exception as e:
        logging.exception(e)
    finally:
        return versions


def list_local_versions(ai_models_dir):
    try:
        model_dirs = [
            {"path": os.path.join(ai_models_dir, f), "version": f}
            for f in os.listdir(ai_models_dir)
            if re.search(r"\d\.\d\.\d", f) and len(os.listdir(os.path.join(ai_models_dir, f))) > 0  # match semantic version
        ]
        return model_dirs
    except Exception as e:
        logging.exception(e)
        return None


def latest_version(ai_models_dir, bucket_name):
    try:
        remote_versions = list_remote_versions(bucket_name)
    except Exception as e:
        remote_versions = []
        logging.exception(e)
    try:
        local_versions = list_local_versions(ai_models_dir)
    except Exception as e:
        local_versions = []
        logging.exception(e)
    ai_options = set([])
    ai_options.update([rv["version"] for rv in remote_versions])
    ai_options.update(set([lv["version"] for lv in local_versions]))
    ai_options = sorted(ai_options, key=lambda k: version.parse(k), reverse=True)
    return ai_options[0]


def ai_model_path_from_version(ai_models_dir, local_version):
    if local_version is None:
        return None

    return os.path.join(ai_models_dir, local_version, "model.onnx")


def compute_orphaned_local_versions(ai_models_dir):
    remote_versions = list_remote_versions(ai_models_dir)

    if remote_versions is None:
        logging.warning("Could not fetch remote versions. Thus, aborting cleaning of local versions in {}. Consider manual cleaning".format(ai_models_dir))
        return

    local_versions = list_local_versions()

    if local_versions is None:
        logging.warning("Could not read local versions in {}. Thus, aborting cleaning. Consider manual cleaning".format(ai_models_dir))
        return

    orphaned_local_versions = [{"path": lv["path"], "version": lv["version"]} for lv in local_versions if lv["version"] not in [rv["version"] for rv in remote_versions]]

    return orphaned_local_versions


def cleanup_orphaned_local_versions(orphaned_local_versions):
    for olv in orphaned_local_versions:
        try:
            shutil.rmtree(olv["path"])
        except Exception as e:
            logging.exception(e)


def download_version(ai_models_dir, bucket_name, target_version, progress=None):
    try:
        remote_versions = list_remote_versions(bucket_name)
        for r in remote_versions:
            if target_version == r["version"]:
                remote_version = r
                break

        ai_model_dir = os.path.join(ai_models_dir, "{}".format(remote_version["version"]))
        os.makedirs(ai_model_dir, exist_ok=True)

        ai_model_file = os.path.join(ai_model_dir, "model.onnx")
        ai_model_zip = os.path.join(ai_model_dir, "model.zip")
        client.fget_object(
            remote_version["bucket"],
            remote_version["object"],
            ai_model_zip,
            progress=DynamicProgressThread(callback=progress),
        )

        with zipfile.ZipFile(ai_model_zip, "r") as zip_ref:
            zip_ref.extractall(ai_model_dir)

        if not os.path.isfile(ai_model_file):
            raise ValueError(f"Could not find ai 'model.onnx' file after extracting {ai_model_zip}")
        os.remove(ai_model_zip)

    except Exception as e:
        # try to delete (rollback) ai_model_dir in case of errors
        logging.exception(e)
        try:
            shutil.rmtree(ai_model_dir)
        except Exception as e2:
            logging.exception(e2)


def validate_local_version(ai_models_dir, local_version):
    return os.path.isfile(os.path.join(ai_models_dir, local_version, "model.onnx"))


def get_execution_providers_ordered(gpu_acceleration=True):

    if gpu_acceleration:
        supported_providers = [
            "DmlExecutionProvider",
            ('CoreMLExecutionProvider', {
                'flags': "COREML_FLAG_CREATE_MLPROGRAM",
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    else:
        supported_providers = ["CPUExecutionProvider"]

    result = []
    for provider in supported_providers:
        if isinstance(provider, tuple):
            if provider[0] in ort.get_available_providers():
                result.append(provider)  # Append the entire tuple
        else:
            if provider in ort.get_available_providers():
                result.append(provider)
    return result


class InferenceEngine:
    def __init__(self, model_path, prefs, auxiliary_model_path=None):
        self.model_path = model_path
        self.aux_model_path = auxiliary_model_path
        self.model_version = os.path.basename(os.path.dirname(model_path))
        self.model_type = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        
        self.patch_size = None
        self.stride = None
        self.channels = None
        self.residuals = None
        self.batch_size = prefs.ai_batch_size

        self.providers = get_execution_providers_ordered(prefs.ai_gpu_acceleration)
        logging.info(f"Available inference providers : {self.providers}")
        
        self.model = None
        self.aux_model = None
        self.model_normalization = None
        self.params_normalization = None
    
    def cleanup(self):
        self.model = None
        self.aux_model = None
        self.model_normalization = None
        self.params_normalization = None

    def load_model(self):
        self.model = ort.InferenceSession(self.model_path, providers=self.providers)

        if self.aux_model_path is not None:
            self.aux_model = ort.InferenceSession(self.aux_model_path, providers=self.providers)

    def load_normalization(self, normalization_dict):
        # Extract major.minor version without patch number
        version_parts = self.model_version.split('.')[:2]
        version_key = '.'.join(version_parts)
        self.model_normalization = normalization_dict[version_key][0]
        self.params_normalization = normalization_dict[version_key][1]

        self.patch_size = normalization_dict[version_key][2].get("patch_size")
        self.stride = normalization_dict[version_key][2].get("stride")
        self.channels = normalization_dict[version_key][2].get("channels")
        self.residuals = normalization_dict[version_key][2].get("residuals")
        self.channel_last = normalization_dict[version_key][2].get("channel_last")

    def calc_best_batch_size(self, num_patches, progress=None, batch_sizes=[1, 2, 4, 8, 16]):
        cancel_flag = False
        def cancel_listener(event):
            nonlocal cancel_flag
            cancel_flag = True
        eventbus.add_listener(AppEvents.CANCEL_PROCESSING, cancel_listener)

        test_data = np.random.rand(num_patches, self.channels, self.patch_size, self.patch_size).astype(np.float32)
        
        best_time = float('inf')
        best_batch_size = 1
        
        # Test different batch sizes
        for idx, batch_size in enumerate(batch_sizes):
            try:
                start_time = time.time()
                last_progress = 0
                
                # Process all patches in batches
                for i in range(0, num_patches, batch_size):
                    if cancel_flag:
                        logging.info("Best batch size calculation cancelled")
                        eventbus.remove_listener(AppEvents.CANCEL_PROCESSING, cancel_listener)
                        return None

                    end_idx = min(i + batch_size, num_patches)
                    batch = test_data[i:end_idx]
                    _ = self.model.run(None, {"gen_input_image": batch, "params": np.zeros((batch_size, 2), dtype=np.float32)})
                    
                    p = int(i / (len(batch_sizes) * num_patches) * 100)
                    if p > last_progress:
                        if progress is not None:
                            progress.update(p - last_progress)
                        else:
                            logging.info(f"Progress: {p}%")
                        last_progress = p
                
                elapsed_time = time.time() - start_time
                
                # Update best batch size if this one was faster
                if elapsed_time < best_time:
                    best_time = elapsed_time
                    best_batch_size = batch_size
                    
            except Exception as e:
                print(e)
                break
        
        return best_batch_size

    def execute(self, image, params, progress=None):
        h, w, c = image.shape

        cancel_flag = False
        def cancel_listener(event):
            nonlocal cancel_flag
            cancel_flag = True
        eventbus.add_listener(AppEvents.CANCEL_PROCESSING, cancel_listener)
        
        num_h = int(np.ceil((h - self.patch_size) / self.stride)) + 1
        num_w = int(np.ceil((w - self.patch_size) / self.stride)) + 1
        
        pad_h = (num_h - 1) * self.stride + self.patch_size - h
        pad_w = (num_w - 1) * self.stride + self.patch_size - w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        new_h = pad_top + h + pad_bottom
        new_w = pad_left + w + pad_right
        
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='reflect'
        )

        patches = np.lib.stride_tricks.sliding_window_view(
            padded_image,
            (self.patch_size, self.patch_size, c)
        )[::self.stride, ::self.stride, 0] 
        patches = patches.reshape(-1, self.patch_size, self.patch_size, c)  #[num_patches, patch_size, patch_size, c]

        patches, norm_params = self.model_normalization.normalize(patches)

        patches = np.moveaxis(patches, -1, 1)   #[num_patches, c, patch_size, patch_size]
        total_patches = patches.shape[0] * (c // self.channels)
        patches = patches.reshape(total_patches, self.channels, self.patch_size, self.patch_size)   #[total_patches, channels, patch_size, patch_size]

        if self.channel_last:
            patches = np.moveaxis(patches, 1, -1)
        
        if params is not None:
            params = self.params_normalization.normalize(params, model_type=self.model_type)

            if params.shape[0] == 1:
                params = np.repeat(params, total_patches, axis=0)

        last_progress = 0
        for idx in range(0, total_patches, self.batch_size):
            if cancel_flag:
                logging.info("AI Inference cancelled")
                eventbus.remove_listener(AppEvents.CANCEL_PROCESSING, cancel_listener)
                return None
            
            model_inputs = {"gen_input_image": patches[idx:idx+self.batch_size]}
            if params is not None:
                model_inputs["params"] = params[idx:idx+self.batch_size]
            
            output = self.model.run(None, model_inputs)[0]

            if self.residuals:
                patches[idx:idx+self.batch_size] = patches[idx:idx+self.batch_size] - output
            else:
                patches[idx:idx+self.batch_size] = output

            p = int(idx / total_patches * 100)
            if p > last_progress:
                if progress is not None:
                    progress.update(p - last_progress)
                else:
                    logging.info(f"Progress: {p}%")
                last_progress = p

        if self.channel_last:
            patches = np.moveaxis(patches, -1, 1)   #[total_patches, channels, patch_size, patch_size]
        
        patches = patches.reshape(total_patches // (c // self.channels), c, self.patch_size, self.patch_size)
        patches = np.moveaxis(patches, 1, -1)   #[num_patches, self.patch_size, self.patch_size, c]

        patches = self.model_normalization.denormalize(patches, norm_params)

        patches = patches.reshape(num_h, num_w, self.patch_size, self.patch_size, c)
        
        reconstructed_image = np.zeros((new_h, new_w, c), dtype=np.float32)
        weights = np.zeros((new_h, new_w, c), dtype=np.float32)

        # TODO : parallelize the following code with numba
        for i in range(num_h):
            for j in range(num_w):
                start_h = i * self.stride
                start_w = j * self.stride

                reconstructed_image[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size] += patches[i, j]
                weights[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size] += 1

        reconstructed_image = reconstructed_image[pad_top:pad_top + h, pad_left:pad_left + w, :]
        weights = weights[pad_top:pad_top + h, pad_left:pad_left + w, :]

        reconstructed_image = reconstructed_image / weights

        eventbus.remove_listener(AppEvents.CANCEL_PROCESSING, cancel_listener)

        return reconstructed_image

    def execute_full_image(self, image, params):
        image, norm_params = self.model_normalization.normalize(image)

        model_inputs = {"gen_input_image": image}
        if params is not None:
            model_inputs["params"] = params

        output = self.model.run(None, model_inputs)[0]

        output = self.model_normalization.denormalize(output, norm_params)

        return output
