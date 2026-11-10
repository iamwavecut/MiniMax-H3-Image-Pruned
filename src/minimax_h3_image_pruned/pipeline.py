"""Load the pruned MiniMax-H3-Image into the Modular Diffusers pipeline."""

from __future__ import annotations

import torch

from .modeling import MiniMaxH3ImagePrunedTransformer2DModel
from .text_encoder import load_text_encoder

DEFAULT_REPO = "WaveCut/MiniMax-H3-Image-Pruned"


def load_transformer(pretrained_model_name_or_path=DEFAULT_REPO, subfolder: str = "transformer",
                     dtype=torch.bfloat16, **kwargs) -> MiniMaxH3ImagePrunedTransformer2DModel:
    return MiniMaxH3ImagePrunedTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, torch_dtype=dtype, **kwargs
    )


def load_pipeline(pretrained_model_name_or_path=DEFAULT_REPO, *, revision: str | None = None, dtype=torch.bfloat16,
                  device: str = "cuda", offload: bool = True, components_manager=None):
    """The `MiniMaxH3ImageModularPipeline` with the pruned transformer and text encoder.

    offload: keep every component in host memory and move each to `device` for its own stage (diffusers'
        `ComponentsManager.enable_auto_cpu_offload`). Without it everything is moved to `device`: about 94 GB.
    """
    from pathlib import Path

    from diffusers import ComponentsManager, ModularPipeline

    manager = components_manager or ComponentsManager()
    if offload:
        manager.enable_auto_cpu_offload(device=device)
    pipe = ModularPipeline.from_pretrained(
        pretrained_model_name_or_path, revision=revision, components_manager=manager
    )
    names = ["vae", "tokenizer", "processor", "scheduler"]
    load_kwargs = {"dtype": dtype, "revision": revision}
    if Path(str(pretrained_model_name_or_path)).is_dir():
        # A local copy: its modular_model_index.json still names the Hub repository.
        load_kwargs["pretrained_model_name_or_path"] = str(pretrained_model_name_or_path)
    pipe.load_components(names=names, **load_kwargs)
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"could not load {missing} from {pretrained_model_name_or_path}; see the warnings above")
    pipe.update_components(
        transformer=load_transformer(pretrained_model_name_or_path, dtype=dtype, revision=revision),
        text_encoder=load_text_encoder(pretrained_model_name_or_path, dtype=dtype, revision=revision),
    )
    if not offload:
        pipe.to(device)
    return pipe
