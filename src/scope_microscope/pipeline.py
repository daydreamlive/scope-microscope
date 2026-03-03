"""Microscope pipeline — CoreML real-time style transfer.

7-stage pipeline ported from microscope/src/pipeline.cpp:
  1. Preprocess: THWC uint8 [0,255] -> center crop -> resize -> NCHW float16 [-1,1]
  2. VAE encode: CoreML taesd_encoder
  3. Latent noise: latent feedback + forward diffusion
  4. UNet predict: CoreML unet
  5. Denoise: reverse diffusion
  6. VAE decode: CoreML taesd_decoder
  7. Postprocess: NCHW float16 [-1,1] -> THWC float32 [0,1]
"""

import logging
import platform
import sys
from typing import TYPE_CHECKING

import numpy as np

from .convert import MODEL_CONFIGS, ensure_models_converted
from .noise_schedule import compute_noise_params
from .schema import MicroscopeConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

from scope.core.pipelines.interface import Pipeline

logger = logging.getLogger(__name__)


class MicroscopePipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return MicroscopeConfig

    def __init__(
        self,
        model_type: str = "sdxs",
        render_size: int = 512,
        strength: float = 0.5,
        **kwargs,
    ):
        if platform.system() != "Darwin":
            raise RuntimeError("MicroscopePipeline requires macOS with Apple Silicon (CoreML)")

        import coremltools as ct
        from transformers import CLIPTokenizer

        self.model_type = model_type
        self.render_size = render_size
        self.latent_size = render_size // 8

        cfg = MODEL_CONFIGS[model_type]
        self.hidden_size = cfg["hidden_size"]

        # Convert/load CoreML models
        logger.info(f"Loading Microscope pipeline: {model_type} @ {render_size}x{render_size}")
        model_dir = ensure_models_converted(model_type, render_size)

        # Load CoreML models with appropriate compute units
        # VAE (tiny) -> CPU + Neural Engine to free GPU for UNet
        # UNet (large) -> CPU + GPU
        cu_vae = ct.ComputeUnit.CPU_AND_NE
        cu_unet = ct.ComputeUnit.CPU_AND_GPU

        logger.info("Loading text_encoder...")
        self.text_encoder = ct.models.MLModel(
            str(model_dir / "text_encoder.mlmodelc"), compute_units=cu_unet
        )

        logger.info("Loading vae_encoder...")
        self.vae_encoder = ct.models.MLModel(
            str(model_dir / f"taesd_encoder_{render_size}.mlmodelc"),
            compute_units=cu_vae,
        )

        logger.info("Loading vae_decoder...")
        self.vae_decoder = ct.models.MLModel(
            str(model_dir / "taesd_decoder.mlmodelc"), compute_units=cu_vae
        )

        unet_name = cfg["unet_prefix"]
        logger.info(f"Loading unet ({model_type})...")
        self.unet = ct.models.MLModel(
            str(model_dir / f"{unet_name}.mlmodelc"), compute_units=cu_unet
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg["model_id"], subfolder="tokenizer"
        )

        # Compute noise schedule
        noise = compute_noise_params(model_type, strength)
        self.timestep = noise.timestep
        self.sqrt_alpha = noise.sqrt_alpha
        self.sqrt_one_minus_alpha = noise.sqrt_one_minus_alpha
        self.inv_sqrt_alpha = noise.inv_sqrt_alpha
        logger.info(
            f"Noise schedule: t={noise.timestep}, alpha={noise.alpha_cumprod:.6f}, "
            f"sqrt_a={noise.sqrt_alpha:.6f}, sqrt_1ma={noise.sqrt_one_minus_alpha:.6f}"
        )

        # Generate fixed noise (seed 42)
        rng = np.random.RandomState(42)
        self.fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(
            np.float16
        )

        # State
        self.prev_denoised: np.ndarray | None = None
        self._prompt_cache: dict[str, np.ndarray] = {}
        self._current_prompt_embeds: np.ndarray | None = None

        # Encode default empty prompt
        self._encode_prompt("")

        logger.info(
            f"Microscope pipeline ready: {render_size}x{render_size} -> "
            f"latent {self.latent_size}x{self.latent_size}"
        )

    def _encode_prompt(self, text: str) -> np.ndarray:
        """Encode a text prompt using CLIP, with caching."""
        if text in self._prompt_cache:
            self._current_prompt_embeds = self._prompt_cache[text]
            return self._current_prompt_embeds

        logger.info(f'Encoding prompt: "{text}"')
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.float16)

        result = self.text_encoder.predict({"input_ids": input_ids})
        embeds = result["last_hidden_state"].astype(np.float16)

        self._prompt_cache[text] = embeds
        self._current_prompt_embeds = embeds
        return embeds

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """THWC uint8 [0,255] -> center crop -> resize -> NCHW float16 [-1,1].

        Args:
            frame: (1, H, W, C) uint8 array

        Returns:
            (1, 3, render_size, render_size) float16 array in [-1, 1]
        """
        from PIL import Image

        # Remove batch dim: (H, W, C)
        img = frame[0]
        h, w = img.shape[:2]
        rs = self.render_size

        # Center crop to square
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        img = img[y0 : y0 + side, x0 : x0 + side]

        # Resize to render_size
        if img.shape[0] != rs or img.shape[1] != rs:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((rs, rs), Image.BILINEAR)
            img = np.array(pil_img)

        # HWC uint8 -> NCHW float16 [-1, 1]
        img_f = img.astype(np.float16) / 127.5 - 1.0
        img_f = img_f.transpose(2, 0, 1)  # CHW
        img_f = img_f[np.newaxis, :3, :, :]  # NCHW, take only RGB
        return img_f

    def _vae_encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to latent space using TAESD encoder."""
        result = self.vae_encoder.predict({"image": image})
        return result["latent"].astype(np.float16)

    def _latent_noise(
        self, latent: np.ndarray, latent_feedback: float
    ) -> np.ndarray:
        """Apply latent feedback and forward diffusion.

        Args:
            latent: Clean latent from VAE encoder
            latent_feedback: Blend factor for previous denoised latent

        Returns:
            Noisy latent for UNet input
        """
        # Latent feedback: blend with previous denoised
        if self.prev_denoised is not None and latent_feedback > 0.0:
            fb = np.float16(latent_feedback)
            latent = (1.0 - fb) * latent + fb * self.prev_denoised

        # Forward diffusion: noisy = sqrt_alpha * clean + sqrt_one_minus_alpha * noise
        sa = np.float16(self.sqrt_alpha)
        s1ma = np.float16(self.sqrt_one_minus_alpha)
        noisy = sa * latent + s1ma * self.fixed_noise
        return noisy.astype(np.float16)

    def _unet_predict(self, noisy: np.ndarray) -> np.ndarray:
        """Run UNet noise prediction."""
        timestep = np.array([float(self.timestep)], dtype=np.float16)
        result = self.unet.predict(
            {
                "sample": noisy,
                "timestep": timestep,
                "encoder_hidden_states": self._current_prompt_embeds,
            }
        )
        return result["noise_pred"].astype(np.float16)

    def _denoise(self, noisy: np.ndarray, noise_pred: np.ndarray) -> np.ndarray:
        """Reverse diffusion step.

        denoised = (noisy - sqrt_one_minus_alpha * noise_pred) * inv_sqrt_alpha
        """
        s1ma = np.float16(self.sqrt_one_minus_alpha)
        inv_sa = np.float16(self.inv_sqrt_alpha)
        denoised = (noisy - s1ma * noise_pred) * inv_sa
        self.prev_denoised = denoised.astype(np.float16)
        return denoised

    def _vae_decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent to image using TAESD decoder."""
        result = self.vae_decoder.predict({"latent": latent})
        return result["image"].astype(np.float16)

    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        """NCHW float16 [-1,1] -> THWC float32 [0,1].

        Args:
            image: (1, 3, H, W) float16 array in [-1, 1]

        Returns:
            (1, H, W, 3) float32 array in [0, 1]
        """
        # [-1, 1] -> [0, 1]
        img = (image.astype(np.float32) + 1.0) * 0.5
        img = np.clip(img, 0.0, 1.0)
        # NCHW -> NHWC
        img = img.transpose(0, 2, 3, 1)
        return img

    def __call__(self, **kwargs) -> dict:
        """Process a video frame through the 7-stage pipeline.

        Args:
            video: List of tensors, each (1, H, W, C) THWC uint8 [0, 255]
            prompts: List of prompt dicts with "text" key
            strength: Denoising strength (runtime, only affects sd-turbo noise schedule)
            latent_feedback: Blend factor for temporal coherence
            init_cache: If True, reset latent feedback state

        Returns:
            Dict with "video" key containing (1, H, W, 3) float32 [0, 1] tensor
        """
        video = kwargs.get("video", [])
        prompts = kwargs.get("prompts")
        latent_feedback = kwargs.get("latent_feedback", 0.0)
        init_cache = kwargs.get("init_cache", False)

        if init_cache:
            self.prev_denoised = None

        # Handle prompt
        if prompts and len(prompts) > 0:
            first_prompt = prompts[0]
            text = (
                first_prompt["text"]
                if isinstance(first_prompt, dict)
                else first_prompt
            )
            self._encode_prompt(text)

        if not video:
            # Return black frame if no input
            rs = self.render_size
            return {"video": np.zeros((1, rs, rs, 3), dtype=np.float32)}

        # Get the first frame as numpy
        frame = video[0]
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        frame = np.asarray(frame, dtype=np.uint8)

        # 7-stage pipeline
        image = self._preprocess(frame)  # 1. Preprocess
        latent = self._vae_encode(image)  # 2. VAE encode
        noisy = self._latent_noise(latent, latent_feedback)  # 3. Latent noise
        noise_pred = self._unet_predict(noisy)  # 4. UNet predict
        denoised = self._denoise(noisy, noise_pred)  # 5. Denoise
        decoded = self._vae_decode(denoised)  # 6. VAE decode
        output = self._postprocess(decoded)  # 7. Postprocess

        return {"video": output}
