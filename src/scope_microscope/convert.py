"""PyTorch to CoreML model conversion for Microscope.

Ported from microscope/scripts/convert_models.py.
Converts text_encoder, vae_encoder, vae_decoder, and unet to CoreML .mlpackage,
then compiles to .mlmodelc for fast loading.
"""

import gc
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "sdxs": {
        "model_id": "IDKiro/sdxs-512-0.9",
        "hidden_size": 1024,
        "unet_prefix": "unet_sdxs_512",
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "hidden_size": 1024,
        "unet_prefix": "unet_sd_turbo",
    },
}


def _compile_mlpackage(mlpackage_path: str, output_dir: str) -> None:
    """Compile .mlpackage to .mlmodelc for fast loading."""
    logger.info(f"Compiling {mlpackage_path}...")
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir],
        check=True,
        capture_output=True,
    )


def _convert_text_encoder(model_id: str, output_dir: str) -> str:
    """Convert CLIP text encoder to CoreML."""
    import coremltools as ct
    import torch
    from transformers import CLIPTextModel

    compiled_path = os.path.join(output_dir, "text_encoder.mlmodelc")
    if os.path.exists(compiled_path):
        logger.info(f"Text encoder already exists: {compiled_path}")
        return compiled_path

    pkg_path = os.path.join(output_dir, "text_encoder.mlpackage")
    if not os.path.exists(pkg_path):
        logger.info("Converting CLIP text encoder...")
        model = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        model.eval()

        class TextEncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, input_ids):
                return self.encoder(input_ids)[0]

        wrapper = TextEncoderWrapper(model).eval()
        dummy_input = torch.randint(0, 49408, (1, 77))

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy_input)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, 77), dtype=np.float16)],
            outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlmodel.save(pkg_path)
        logger.info(f"Saved: {pkg_path}")

        del mlmodel, traced, wrapper, model
        gc.collect()

    _compile_mlpackage(pkg_path, output_dir)
    return compiled_path


def _convert_vae_encoder(render_size: int, output_dir: str) -> str:
    """Convert TinyVAE encoder to CoreML."""
    import coremltools as ct
    import torch
    from diffusers import AutoencoderTiny

    name = f"taesd_encoder_{render_size}"
    compiled_path = os.path.join(output_dir, f"{name}.mlmodelc")
    if os.path.exists(compiled_path):
        logger.info(f"VAE encoder already exists: {compiled_path}")
        return compiled_path

    pkg_path = os.path.join(output_dir, f"{name}.mlpackage")
    if not os.path.exists(pkg_path):
        logger.info(f"Converting TinyVAE encoder ({render_size}x{render_size})...")
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

        class Wrapper(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.encoder = v.encoder

            def forward(self, x):
                return self.encoder(x)

        wrapper = Wrapper(vae).eval()
        dummy = torch.randn(1, 3, render_size, render_size)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="latent", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlmodel.save(pkg_path)
        logger.info(f"Saved: {pkg_path}")

        del mlmodel, traced, wrapper, vae
        gc.collect()

    _compile_mlpackage(pkg_path, output_dir)
    return compiled_path


def _convert_vae_decoder(render_size: int, output_dir: str) -> str:
    """Convert TinyVAE decoder to CoreML."""
    import coremltools as ct
    import torch
    from diffusers import AutoencoderTiny

    compiled_path = os.path.join(output_dir, "taesd_decoder.mlmodelc")
    if os.path.exists(compiled_path):
        logger.info(f"VAE decoder already exists: {compiled_path}")
        return compiled_path

    pkg_path = os.path.join(output_dir, "taesd_decoder.mlpackage")
    if not os.path.exists(pkg_path):
        latent_size = render_size // 8
        logger.info(f"Converting TinyVAE decoder ({render_size}x{render_size})...")
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

        class Wrapper(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.decoder = v.decoder

            def forward(self, x):
                return self.decoder(x)

        wrapper = Wrapper(vae).eval()
        dummy = torch.randn(1, 4, latent_size, latent_size)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy)

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="latent", shape=dummy.shape, dtype=np.float16)
            ],
            outputs=[ct.TensorType(name="image", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlmodel.save(pkg_path)
        logger.info(f"Saved: {pkg_path}")

        del mlmodel, traced, wrapper, vae
        gc.collect()

    _compile_mlpackage(pkg_path, output_dir)
    return compiled_path


def _convert_unet(
    model_id: str, cfg: dict, render_size: int, output_dir: str
) -> str:
    """Convert UNet to CoreML."""
    import coremltools as ct
    import torch
    from diffusers import StableDiffusionPipeline

    prefix = cfg["unet_prefix"]
    compiled_path = os.path.join(output_dir, f"{prefix}.mlmodelc")
    if os.path.exists(compiled_path):
        logger.info(f"UNet already exists: {compiled_path}")
        return compiled_path

    pkg_path = os.path.join(output_dir, f"{prefix}.mlpackage")
    if not os.path.exists(pkg_path):
        latent_size = render_size // 8
        logger.info(f"Converting UNet ({prefix})...")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        unet = pipe.unet.eval().float().cpu()

        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, sample, timestep, encoder_hidden_states):
                return self.unet(sample, timestep, encoder_hidden_states).sample

        wrapper = UNetWrapper(unet).eval()
        hidden_size = cfg["hidden_size"]
        dummy_sample = torch.randn(1, 4, latent_size, latent_size)
        dummy_timestep = torch.tensor([999.0])
        dummy_hidden = torch.randn(1, 77, hidden_size)

        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper, (dummy_sample, dummy_timestep, dummy_hidden)
            )

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="sample", shape=dummy_sample.shape, dtype=np.float16
                ),
                ct.TensorType(name="timestep", shape=(1,), dtype=np.float16),
                ct.TensorType(
                    name="encoder_hidden_states",
                    shape=dummy_hidden.shape,
                    dtype=np.float16,
                ),
            ],
            outputs=[ct.TensorType(name="noise_pred", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlmodel.save(pkg_path)
        logger.info(f"Saved: {pkg_path}")

        del mlmodel, traced, wrapper, unet, pipe
        gc.collect()

    _compile_mlpackage(pkg_path, output_dir)
    return compiled_path


def ensure_models_converted(model_type: str, render_size: int = 512) -> Path:
    """Convert all required models if not already cached.

    Args:
        model_type: "sdxs" or "sd-turbo"
        render_size: Render resolution (default 512)

    Returns:
        Path to the model cache directory containing .mlmodelc files
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Expected: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_type]
    model_id = cfg["model_id"]

    cache_dir = Path.home() / ".daydream-scope" / "models" / "microscope" / f"{model_type}_{render_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(cache_dir)

    logger.info(f"Ensuring models for {model_type} ({render_size}x{render_size}) in {cache_dir}")

    _convert_text_encoder(model_id, output_dir)
    _convert_vae_encoder(render_size, output_dir)
    _convert_vae_decoder(render_size, output_dir)
    _convert_unet(model_id, cfg, render_size, output_dir)

    logger.info(f"All models ready in {cache_dir}")
    return cache_dir
