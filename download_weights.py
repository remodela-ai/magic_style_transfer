import logging
import os, shutil, torch
from diffusers import (
    AutoencoderKL, 
    DiffusionPipeline,
    ControlNetModel
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import (
    DPTFeatureExtractor, 
    DPTForDepthEstimation, 
    CLIPVisionModelWithProjection
)


# Setup logging
logging.basicConfig(level=logging.INFO)

def setup_directory(folder):
    """Create a new folder or clear an existing one."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

def download_and_save_model(model_class, model_name, save_path, **kwargs):
    """Download and save a pretrained model."""
    logging.info(f"Downloading {model_name}...")
    try:
        model = model_class.from_pretrained(model_name, **kwargs)
        model.save_pretrained(save_path,safe_serialization=True)
        logging.info(f"Model {model_name} saved to {save_path}.")
    except Exception as e:
        logging.error(f"Failed to download {model_name}: {e}")

# Create directories for caching models
CACHE_DIRS = {
    "CONTROL_DEPTH_CACHE": setup_directory("control-depth-cache"),
    "CONTROL_CANNY_CACHE": setup_directory("control-canny-cache"),
    "SDXL_MODEL_CACHE": setup_directory("sdxl-cache"),
    "SAFETY_CACHE": setup_directory("safety-cache"),
    "FEATURE_CACHE": setup_directory("feature-cache"),
    "FEATURE_EXTRACTOR": setup_directory("feature-extractor"),
    "CLIP_CACHE": setup_directory("clip-cache"),
}


download_and_save_model(
    DiffusionPipeline, 
    "stabilityai/stable-diffusion-xl-base-1.0", 
    CACHE_DIRS["SDXL_MODEL_CACHE"], 
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)


download_and_save_model(
    ControlNetModel, 
    "diffusers/controlnet-depth-sdxl-1.0", 
    CACHE_DIRS["CONTROL_DEPTH_CACHE"],
    torch_dtype=torch.float16
)

download_and_save_model(
    ControlNetModel, 
    "diffusers/controlnet-canny-sdxl-1.0", 
    CACHE_DIRS["CONTROL_CANNY_CACHE"],
    torch_dtype=torch.float16
)



download_and_save_model(
    DPTForDepthEstimation, 
    "Intel/dpt-hybrid-midas", 
    CACHE_DIRS["FEATURE_CACHE"], 
    torch_dtype=torch.float16
)

download_and_save_model(
    DPTFeatureExtractor, 
    "Intel/dpt-hybrid-midas", 
    CACHE_DIRS["FEATURE_EXTRACTOR"]
)


download_and_save_model(
    StableDiffusionSafetyChecker, 
    "CompVis/stable-diffusion-safety-checker", 
    CACHE_DIRS["SAFETY_CACHE"], 
    torch_dtype=torch.float16
)

download_and_save_model(
    CLIPVisionModelWithProjection, 
    "h94/IP-Adapter",
    CACHE_DIRS["CLIP_CACHE"], 
    torch_dtype=torch.float16,
    subfolder="models/image_encoder",    

)

