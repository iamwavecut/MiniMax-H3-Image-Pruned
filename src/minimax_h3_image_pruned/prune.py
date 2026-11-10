"""Build the pruned MiniMax-H3-Image checkpoint from the official release.

    minimax-h3-image-prune --output ./MiniMax-H3-Image-Pruned [--source ./MiniMax-H3-Image] [--device cuda]

Without `--source` the needed files of `MiniMaxAI/MiniMax-H3-Image` are downloaded first (about 135 GB). The output is a
Modular Diffusers repository:

* `transformer/`: the diffusers transformer without `transformer_blocks.*.adaln_proj.linear`, `time_embedder` and
  `norm_out.linear`, plus the modulation tables that replace them. Every kept tensor is a byte copy.
* `text_encoder/`: the Qwen3-VL conditioner without decoder layers 50-63 and `lm_head`. The pipeline reads
  `hidden_states[50]`, the output of layer 49, so those weights never contribute.
* `vae/`, `tokenizer/`, `processor/`, `scheduler/`: unchanged copies.

The tables are computed on `--device` with the original modules and the matrix shapes the pipeline uses. Build them on
the GPU type you run on to make the pruned transformer bit-identical to the original one there; on another GPU type the
two differ by the original model's own cross-device rounding (single bfloat16 ulps in a few table entries).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .modeling import NUM_TAGS
from .safetensors_io import NewTensor, read_checkpoint, write_sharded

SOURCE_REPO = "MiniMaxAI/MiniMax-H3-Image"
SOURCE_PATTERNS = [
    "transformer_diffusers/*",
    "text_encoder/*",
    "vae_diffusers/*",
    "tokenizer/*",
    "processor/*",
    "scheduler/*",
    "modular_model_index.json",
    "LICENSE",
]
TEXT_ENCODER_LAYER = 50  # the transformer is conditioned on `hidden_states[50]`
NOTICE_LINE = (
    "MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, "
    "Copyright © 2026 MiniMax. All Rights Reserved."
)
MODULE_FILE = "modeling_minimax_h3_image_pruned.py"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def modification_notice(what: str, source_revision: str | None) -> str:
    rev = f"@{source_revision}" if source_revision else ""
    return f"Modified by WaveCut from {SOURCE_REPO}{rev}: {what}"


# ---------------------------------------------------------------------------------------------------------------------
# Timestep sets


def reference_timestep() -> float:
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import MiniMaxH3ImageModularPipeline

    return float(MiniMaxH3ImageModularPipeline.reference_noise_aug.fget(None))


def timestep_plans(source: Path) -> list[list[float]]:
    """The `timestep` tensors the pipeline passes, as `MiniMaxH3ImageSetTimestepsStep` builds them.

    One set per text-to-image step (the step value) and one per edit step (the step value and the reference value).
    """
    from diffusers import MiniMaxH3Scheduler

    sample_sigmas = json.loads((source / "modular_model_index.json").read_text())["sample_sigmas"]
    scheduler = MiniMaxH3Scheduler.from_pretrained(str(source / "scheduler"))
    timesteps = torch.tensor([1.0 - sigma for sigma in sample_sigmas[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    reference = reference_timestep()
    plans = []
    for with_reference in (False, True):
        for timestep in scheduler.timesteps:
            rows = torch.full((2,), float(timestep), dtype=torch.float32)
            if with_reference:
                rows[0] = reference
            plans.append([float(v) for v in torch.unique(rows, sorted=True).tolist()])
    return plans


# ---------------------------------------------------------------------------------------------------------------------
# Modulation tables


class _Reader:
    """Tensor access by name across the shards of a checkpoint folder."""

    def __init__(self, folder: Path):
        from safetensors import safe_open

        self._open = safe_open
        index = json.loads(next(folder.glob("*.safetensors.index.json")).read_text())["weight_map"]
        self._map = {name: folder / shard for name, shard in index.items()}
        self._files = {}

    def get(self, name: str) -> torch.Tensor:
        path = self._map[name]
        if path not in self._files:
            self._files[path] = self._open(str(path), framework="pt")
        return self._files[path].get_tensor(name)


@torch.no_grad()
def build_tables(transformer_dir: Path, plans: list[list[float]], device: str):
    """`(block_tables, norm_out_table)`: what `adaln_proj` and `norm_out.linear` return for every timestep set.

    Mirrors `MiniMaxH3ImageTransformer2DModel.forward` loaded in bfloat16: `time_proj` and the float32
    `time_embedder` on the `timestep` tensor of the call, then each bfloat16 projection on that `temb`.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3_image import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3ImageAdaLayerNormModulation,
    )

    cfg = json.loads((transformer_dir / "config.json").read_text())
    hidden, temb_dim = cfg["hidden_size"], cfg["time_embed_dim"]
    reader = _Reader(transformer_dir)

    def load(module, prefix, dtype):
        module.load_state_dict({k: reader.get(f"{prefix}.{k}") for k in module.state_dict()})
        return module.to(device=device, dtype=dtype).eval()

    time_proj = Timesteps(num_channels=cfg["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(
        TimestepEmbedding(in_channels=cfg["freq_dim"], time_embed_dim=cfg["time_embed_hidden_dim"], out_dim=temb_dim),
        "time_embedder",
        torch.float32,
    )
    tembs = []
    for plan in plans:
        timestep = torch.tensor(plan, dtype=torch.float32, device=device)
        temb = time_proj(timestep)
        tembs.append(time_embedder(temb.to(get_parameter_dtype(time_embedder))))

    block_tables = []
    modulation = MiniMaxH3ImageAdaLayerNormModulation(time_embed_dim=temb_dim, hidden_size=hidden)
    for i in range(cfg["num_layers"]):
        load(modulation, f"transformer_blocks.{i}.adaln_proj", torch.bfloat16)
        rows = []
        for plan, temb in zip(plans, tembs):
            full = torch.cat(modulation(temb), dim=-1)  # (len(plan) * 3, 6 * hidden), rows [t0 m0, t0 m1, t0 m2, ...]
            full = full.view(len(plan), MINIMAX_H3_MODALITY_NUM, 6 * hidden)[:, :NUM_TAGS]
            rows.append(full.reshape(len(plan) * NUM_TAGS, 6 * hidden))
        block_tables.append(torch.cat(rows).cpu())
        if i % 10 == 9:
            log(f"modulation tables: {i + 1}/{cfg['num_layers']} blocks")

    out_linear = load(torch.nn.Linear(temb_dim, 2 * hidden, bias=True), "norm_out.linear", torch.bfloat16)
    out_rows = [out_linear(torch.nn.functional.silu(temb).to(get_parameter_dtype(out_linear))) for temb in tembs]
    return block_tables, torch.cat(out_rows).cpu()


# ---------------------------------------------------------------------------------------------------------------------
# Components


TRANSFORMER_DROP = re.compile(r"^(transformer_blocks\.\d+\.adaln_proj\.linear\.|time_embedder\.|norm_out\.linear\.)")
TEXT_ENCODER_DROP = re.compile(
    r"^(lm_head\.|model\.language_model\.layers\.(\d+)\.)"
)


def build_transformer(source: Path, output: Path, plans, device: str, revision: str | None) -> None:
    src = source / "transformer_diffusers"
    dst = output / "transformer"
    dst.mkdir(parents=True, exist_ok=True)
    stored = read_checkpoint(src)
    kept = [t for t in stored if not TRANSFORMER_DROP.match(t.name)]
    dropped = sum(t.nbytes for t in stored) - sum(t.nbytes for t in kept)
    log(f"transformer: {len(stored)} tensors, dropping {len(stored) - len(kept)} ({dropped / 1e9:.2f} GB)")

    started = time.perf_counter()
    block_tables, out_table = build_tables(src, plans, device)
    log(f"modulation tables: {sum(t.numel() for t in block_tables) * 2 / 1e6:.1f} MB in "
        f"{time.perf_counter() - started:.0f} s on {device}")
    new = [NewTensor(f"transformer_blocks.{i}.adaln_proj.table", t) for i, t in enumerate(block_tables)]
    new.append(NewTensor("norm_out.table", out_table))

    notice = modification_notice(
        "AdaLN projections, time_embedder and norm_out.linear replaced by precomputed modulation tables.", revision
    )
    write_sharded(dst, "diffusion_pytorch_model", kept + new, {"format": "pt", "notice": notice}, log=log)

    cfg = json.loads((src / "config.json").read_text())
    cfg["_class_name"] = "MiniMaxH3ImagePrunedTransformer2DModel"
    cfg["adaln_plans"] = plans
    cfg["reference_timestep"] = reference_timestep()
    cfg["auto_map"] = {"AutoModel": f"{Path(MODULE_FILE).stem}.MiniMaxH3ImagePrunedTransformer2DModel"}
    cfg["_notice"] = notice
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("modeling.py"), dst / MODULE_FILE)


def build_text_encoder(source: Path, output: Path, revision: str | None) -> None:
    src = source / "text_encoder"
    dst = output / "text_encoder"
    dst.mkdir(parents=True, exist_ok=True)

    def keep(name: str) -> bool:
        m = TEXT_ENCODER_DROP.match(name)
        if m is None:
            return True
        return m.group(2) is not None and int(m.group(2)) < TEXT_ENCODER_LAYER

    stored = read_checkpoint(src)
    kept = [t for t in stored if keep(t.name)]
    dropped = sum(t.nbytes for t in stored) - sum(t.nbytes for t in kept)
    log(f"text encoder: {len(stored)} tensors, dropping {len(stored) - len(kept)} ({dropped / 1e9:.2f} GB)")
    notice = modification_notice(
        f"decoder layers {TEXT_ENCODER_LAYER}+ and lm_head removed; the pipeline reads hidden_states[{TEXT_ENCODER_LAYER}].",
        revision,
    )
    write_sharded(dst, "model", kept, {"format": "pt", "notice": notice}, log=log)

    cfg = json.loads((src / "config.json").read_text())
    cfg["text_config"]["num_hidden_layers"] = TEXT_ENCODER_LAYER
    # `lm_head` is not stored: tie it to the input embedding so that loading allocates and initializes nothing.
    cfg["tie_word_embeddings"] = True
    cfg["_notice"] = notice
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            _copy(f, dst / f.name)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def copy_unchanged(source: Path, output: Path) -> None:
    for src_name, dst_name in (("vae_diffusers", "vae"), ("tokenizer", "tokenizer"), ("processor", "processor"),
                               ("scheduler", "scheduler")):
        for f in sorted((source / src_name).iterdir()):
            if f.is_file():
                _copy(f, output / dst_name / f.name)
    _copy(source / "LICENSE", output / "LICENSE")
    (output / "NOTICE").write_text(
        NOTICE_LINE + "\n\nThis repository is a modified version of MiniMax-H3-Image by WaveCut. Modified files carry a "
        "`_notice` entry (configs) or a `notice` metadata field (safetensors) describing the change.\n"
    )


def write_modular_index(source: Path, output: Path, repo_id: str) -> None:
    index = json.loads((source / "modular_model_index.json").read_text())
    subfolders = {"transformer": "transformer", "vae": "vae"}
    for name, value in index.items():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            spec = value[2]
            spec["pretrained_model_name_or_path"] = repo_id
            spec["subfolder"] = subfolders.get(name, spec.get("subfolder"))
            spec["revision"] = None
    index["transformer"][1] = "AutoModel"
    index["transformer"][2]["type_hint"] = ["diffusers", "AutoModel"]
    (output / "modular_model_index.json").write_text(json.dumps(index, indent=2) + "\n")


def fetch_source(revision: str | None, cache_dir: str | None) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    revision = revision or HfApi().model_info(SOURCE_REPO).sha
    path = snapshot_download(SOURCE_REPO, revision=revision, allow_patterns=SOURCE_PATTERNS, cache_dir=cache_dir)
    return Path(path), revision


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source", type=Path, help="local copy of MiniMaxAI/MiniMax-H3-Image (downloaded if omitted)")
    ap.add_argument("--source-revision", help="revision of the source repository (recorded in the notices)")
    ap.add_argument("--repo-id", default="WaveCut/MiniMax-H3-Image-Pruned",
                    help="repository the output's modular_model_index.json points at")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir")
    ap.add_argument("--skip", nargs="*", default=[], choices=["transformer", "text_encoder", "copies"])
    args = ap.parse_args(argv)

    if args.source is None:
        args.source, args.source_revision = fetch_source(args.source_revision, args.cache_dir)
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    plans = timestep_plans(source)
    log(f"timestep sets: {plans}")
    if "transformer" not in args.skip:
        build_transformer(source, output, plans, args.device, args.source_revision)
    if "text_encoder" not in args.skip:
        build_text_encoder(source, output, args.source_revision)
    if "copies" not in args.skip:
        copy_unchanged(source, output)
        write_modular_index(source, output, args.repo_id)
    sizes = {p.name: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in output.iterdir() if p.is_dir()}
    log("sizes: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sorted(sizes.items())))
    log("PRUNE_DONE")


if __name__ == "__main__":
    main()
