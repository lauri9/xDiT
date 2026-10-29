"""Public entry points.

Importing this package, which Python does before any submodule, used to import every
pipeline. A distributed worker then paid for diffusers and the rest of that stack
before it had asked for a model. Names below load on first access. ``from xfuser
import xFuserFluxPipeline`` still works, and still imports that pipeline.
"""

import importlib

from xfuser.compat import import_optional

__all__ = [
    "xFuserPixArtAlphaPipeline",
    "xFuserPixArtSigmaPipeline",
    "xFuserStableDiffusion3Pipeline",
    "xFuserFluxPipeline",
    "xFuserLattePipeline",
    "xFuserHunyuanDiTPipeline",
    "xFuserCogVideoXPipeline",
    "xFuserConsisIDPipeline",
    "xFuserStableDiffusionXLPipeline",
    "xFuserSanaPipeline",
    "xFuserSanaSprintPipeline",
    "xFuserLumina2Pipeline",
    "xFuserArgs",
    "EngineConfig",
    "xDiTParallel",
]

# (module, attribute). Importing the module is the load; the attribute is the export.
_LAZY_EXPORTS = {
    "xFuserPixArtAlphaPipeline": (
        "xfuser.model_executor.pipelines.pipeline_pixart_alpha",
        "xFuserPixArtAlphaPipeline",
    ),
    "xFuserPixArtSigmaPipeline": (
        "xfuser.model_executor.pipelines.pipeline_pixart_sigma",
        "xFuserPixArtSigmaPipeline",
    ),
    "xFuserStableDiffusion3Pipeline": (
        "xfuser.model_executor.pipelines.pipeline_stable_diffusion_3",
        "xFuserStableDiffusion3Pipeline",
    ),
    "xFuserFluxPipeline": (
        "xfuser.model_executor.pipelines.pipeline_flux",
        "xFuserFluxPipeline",
    ),
    "xFuserLattePipeline": (
        "xfuser.model_executor.pipelines.pipeline_latte",
        "xFuserLattePipeline",
    ),
    "xFuserHunyuanDiTPipeline": (
        "xfuser.model_executor.pipelines.pipeline_hunyuandit",
        "xFuserHunyuanDiTPipeline",
    ),
    "xFuserCogVideoXPipeline": (
        "xfuser.model_executor.pipelines.pipeline_cogvideox",
        "xFuserCogVideoXPipeline",
    ),
    "xFuserConsisIDPipeline": (
        "xfuser.model_executor.pipelines.pipeline_consisid",
        "xFuserConsisIDPipeline",
    ),
    "xFuserStableDiffusionXLPipeline": (
        "xfuser.model_executor.pipelines.pipeline_stable_diffusion_xl",
        "xFuserStableDiffusionXLPipeline",
    ),
    "xFuserSanaPipeline": (
        "xfuser.model_executor.pipelines.pipeline_sana",
        "xFuserSanaPipeline",
    ),
    "xFuserSanaSprintPipeline": (
        "xfuser.model_executor.pipelines.pipeline_sana_sprint",
        "xFuserSanaSprintPipeline",
    ),
    "xFuserLumina2Pipeline": (
        "xfuser.model_executor.pipelines.pipeline_lumina2",
        "xFuserLumina2Pipeline",
    ),
    "xFuserArgs": ("xfuser.config", "xFuserArgs"),
    "EngineConfig": ("xfuser.config", "EngineConfig"),
    "xDiTParallel": ("xfuser.parallel", "xDiTParallel"),
}

# These need diffusers symbols newer than the install floor. Importing the module is
# the availability test, same as optional_exporter used to do at package import.
_OPTIONAL_EXPORTS = {
    "xFuserFlux2Pipeline": "xfuser.model_executor.pipelines.pipeline_flux2",
    "xFuserFlux2KleinPipeline": "xfuser.model_executor.pipelines.pipeline_flux2_klein",
}


def __getattr__(name):
    optional = _OPTIONAL_EXPORTS.get(name)
    if optional is not None:
        module = import_optional(optional)
        if module is None or not hasattr(module, name):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(module, name)
        globals()[name] = value
        return value

    exported = _LAZY_EXPORTS.get(name)
    if exported is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = exported
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_OPTIONAL_EXPORTS))
