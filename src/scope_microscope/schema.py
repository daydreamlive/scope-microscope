"""Configuration schema for Microscope pipeline."""

from typing import ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.artifacts import Artifact, HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    height_field,
    ui_field_config,
    width_field,
)


class MicroscopeConfig(BasePipelineConfig):
    """Configuration for Microscope CoreML real-time style transfer pipeline."""

    pipeline_id = "microscope"
    pipeline_name = "Microscope"
    pipeline_description = "Real-time style transfer using CoreML on Apple Silicon (SDXS / SD-Turbo)"
    pipeline_version = "0.1.0"

    supports_prompts = True
    default_temporal_interpolation_method = None
    default_spatial_interpolation_method = None

    modes = {"video": ModeDefaults(default=True, input_size=1)}

    artifacts: ClassVar[list[Artifact]] = [
        HuggingfaceRepoArtifact(
            repo_id="junhyr/microscope",
            files=[
                "text_encoder.mlpackage",
                "taesd_encoder_512.mlpackage",
                "taesd_decoder.mlpackage",
                "unet_sdxs_512.mlpackage",
                "unet_sd_turbo.mlpackage",
            ],
        ),
    ]

    # Load parameters (require pipeline reload)
    model_type: Literal["sdxs", "sd-turbo"] = Field(
        default="sd-turbo",
        description="Diffusion model variant",
        json_schema_extra=ui_field_config(
            order=0, is_load_param=True, label="Model"
        ),
    )
    render_size: int = Field(
        default=512,
        description="Render resolution (square)",
        json_schema_extra=ui_field_config(
            order=1, is_load_param=True, label="Resolution"
        ),
    )

    # Resolution locked to render_size
    height: int = height_field(512)
    width: int = width_field(512)

    # Runtime parameters
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Denoising strength (only affects sd-turbo)",
        json_schema_extra=ui_field_config(order=10, modes=["video"], label="Strength"),
    )
    latent_feedback: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Blend factor for previous denoised latent (temporal coherence)",
        json_schema_extra=ui_field_config(
            order=11, modes=["video"], label="Latent Feedback"
        ),
    )
