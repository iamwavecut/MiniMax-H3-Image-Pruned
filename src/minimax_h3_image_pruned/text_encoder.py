"""The 50-layer MiniMax-H3 Qwen3-VL conditioner (Transformers).

MiniMax-H3-Image is conditioned on `hidden_states[50]` of its Qwen3-VL text encoder: the output of decoder layer 49,
before the final norm. The pruned checkpoint keeps layers 0-49 only. Loaded as is, the last hidden state of a 50-layer
stack would go through the final norm, so `adapt_text_encoder` replaces that norm with an identity. It then reports one
layer more in the config, because the diffusers pipeline checks that the conditioner is deeper than the layer it reads;
`hidden_states[50]` of the adapted model is exactly the original `hidden_states[50]`.
"""

from __future__ import annotations

import torch
from torch import nn

TEXT_ENCODER_LAYER = 50


def adapt_text_encoder(model, layer: int = TEXT_ENCODER_LAYER):
    """Make a 50-layer `Qwen3VLForConditionalGeneration` (or `Qwen3VLModel`) return the conditioning hidden state."""
    language_model = model.model.language_model if hasattr(model, "lm_head") else model.language_model
    if len(language_model.layers) != layer:
        raise ValueError(f"expected a {layer}-layer conditioner, got {len(language_model.layers)} layers")
    language_model.norm = nn.Identity()
    model.config.text_config.num_hidden_layers = layer + 1
    return model.eval()


def load_text_encoder(pretrained_model_name_or_path, subfolder: str = "text_encoder", dtype=torch.bfloat16, **kwargs):
    """`Qwen3VLForConditionalGeneration` from the pruned checkpoint, adapted.

    Its `lm_head` is tied to the input embedding (the checkpoint does not store it), so it costs no memory; it is not a
    usable language model head. Read the conditioning as
    `model.model(..., output_hidden_states=True).hidden_states[50]` or `.last_hidden_state`.
    """
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, dtype=dtype, **kwargs
    )
    return adapt_text_encoder(model)
