import json
import os
import shutil
import subprocess
import time
import os
import cv2
import torch
import logging
from typing import List, Optional, Tuple, Union
from weights import WeightsDownloadCache
import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from transformers import CLIPVisionModelWithProjection
from dataset_and_utils import TokenEmbeddingsHandler

logging.basicConfig(level=logging.INFO)


CONTROL_DEPTH_CACHE="control-depth-cache"
CONTROL_CANNY_CACHE="control-canny-cache"
SDXL_MODEL_CACHE="sdxl-cache"
SAFETY_CACHE="safety-cache"
FEATURE_CACHE="feature-cache"
FEATURE_EXTRACTOR="feature-extractor"
CLIP_CACHE="clip-cache"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Predictor(BasePredictor):

    def load_lora(self, weights, pipe, scale):
        logging.info("Loading Unet LoRA")
        self.is_lora = True
        weights = str(weights)
        self.tuned_weights = weights
        local_weights_cache = self.weights_cache.ensure(weights)
        self.path = os.path.join(local_weights_cache, "lora.safetensors")
        pipe.load_lora_weights(self.path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[scale])
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params
        self.tuned_model = True
        

    def unload_lora(self, pipe):
        pipe.unload_lora_weights()
        self.tuned_model = False


    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        self.weights_cache = WeightsDownloadCache()

        logging.info("Loading safety checker...")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")

        self.depth_estimator = DPTForDepthEstimation.from_pretrained(FEATURE_CACHE).to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR)

        controlnet = [
            ControlNetModel.from_pretrained(
                CONTROL_DEPTH_CACHE,
                torch_dtype=torch.float16,
            ),
            ControlNetModel.from_pretrained(
                CONTROL_CANNY_CACHE,
                torch_dtype=torch.float16,
            )
        ]

        self.image_estimator = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_CACHE,
            torch_dtype=torch.float16
        )

        logging.info("Loading SDXL Controlnet pipeline...")

        self.control_img2img_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            controlnet=controlnet,
            image_encoder=self.image_estimator,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.control_img2img_pipe.to("cuda")
        
        self.control_img2img_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")

        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_lora(weights, self.control_img2img_pipe)

        logging.info("setup took: ", time.time() - start)

    @staticmethod
    def load_image(path, color="#A2A2A2", resizing=1):
        shutil.copyfile(path, "/tmp/image.png")
        img = Image.open("/tmp/image.png")
        original_size = img.size
        background_size = (int(original_size[0] * resizing), int(original_size[1] * resizing))
        background = Image.new('RGB', background_size, color)
        position = ((background_size[0] - original_size[0]) // 2, (background_size[1] - original_size[1]) // 2)
        if img.mode == 'RGBA':
            transparent_overlay = Image.new('RGBA', background_size, (0, 0, 0, 0))
            transparent_overlay.paste(img, position, img)
            img = Image.alpha_composite(background.convert('RGBA'), transparent_overlay).convert('RGB')
        else:
            background.paste(img, position)
            img = background
        return img
    
    def resize_image(self, image):
        image_width, image_height = image.size
        logging.info("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        logging.info("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height
    
    @staticmethod   
    def resize_to_allowed_dimensions(width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        logging.info(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions
    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        height, width = image.shape[2], image.shape[3]

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image
    @staticmethod   
    def image2canny(image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        image: Path = Input(
            description="Input image",
            default=None,
        ),
        ip_image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        condition_depth_scale: float = Input(
            description="The bigger this number is, the more ControlNet interferes",
            default=0.35,
            ge=0.0,
            le=2.0,
        ),
        condition_canny_scale: float = Input(
            description="The bigger this number is, the more ControlNet interferes",
            default=0.15,
            ge=0.0,
            le=2.0,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        ip_scale: float = Input(
            description="IP Adapter strength.",
            ge=0.0,
            le=1.0,
            default=0.3,
        ),
        strength: float = Input(
            description="When img2img is active, the denoising strength. 1 means total destruction of the input image.",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        background_color: str = Input(
            description="When passing an image with alpha channel, it will be replaced with this color",
            default="#A2A2A2"
        ),
        resizing_scale: float = Input(
            description="If you want the image to have a solid margin. Scale of the solid margin. 1.0 means no resizing.",
            default=1.0,
            ge = 1.0,
            le = 10.0
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        logging.info(f"Using seed: {seed}")
        
        pipe = self.control_img2img_pipe
        # Sometimes loras are not properly unloaded at the end of the execution, try again here.
        self.unload_lora(pipe)
        
        pipe.set_ip_adapter_scale(ip_scale)
        
        if lora_weights:
            self.load_lora(lora_weights, pipe, lora_scale)

        # OOMs can leave vae in bad state
        if self.control_img2img_pipe.vae.dtype == torch.float32:
            self.control_img2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        logging.info(f"Prompt: {prompt}")
        image = self.load_image(image, background_color, resizing_scale)
        ip_image = self.load_image(ip_image, background_color, resizing_scale)
        resized_image, width, height = self.resize_image(image)

        sdxl_kwargs["image"] = resized_image
        sdxl_kwargs["control_image"] = [self.get_depth_map(image), self.image2canny(image)]
        sdxl_kwargs["ip_adapter_image"] = ip_image
        sdxl_kwargs["strength"] = strength
        sdxl_kwargs["controlnet_conditioning_scale"] = [condition_depth_scale, condition_canny_scale]
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1}

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache

        output_paths = []
        for i, nsfw in enumerate(output):
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
        
        self.unload_lora(pipe)

        return output_paths
