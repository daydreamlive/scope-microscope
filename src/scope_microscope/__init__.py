"""Microscope plugin for Daydream Scope."""

import scope.core

from .pipeline import MicroscopePipeline


@scope.core.hookimpl
def register_pipelines(register):
    register(MicroscopePipeline)


__all__ = ["MicroscopePipeline"]
