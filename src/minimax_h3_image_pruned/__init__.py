"""MiniMax-H3-Image with tabulated AdaLN modulation and a 50-layer text encoder."""

from .modeling import MiniMaxH3ImagePrunedTransformer2DModel
from .pipeline import DEFAULT_REPO, load_pipeline, load_transformer
from .text_encoder import adapt_text_encoder, load_text_encoder

__all__ = [
    "DEFAULT_REPO",
    "MiniMaxH3ImagePrunedTransformer2DModel",
    "adapt_text_encoder",
    "load_pipeline",
    "load_text_encoder",
    "load_transformer",
]
__version__ = "0.1.0"
