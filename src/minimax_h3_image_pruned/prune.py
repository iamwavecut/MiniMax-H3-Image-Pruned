"""Build the pruned MiniMax-H3-Image checkpoint from the official release.

    minimax-h3-image-prune --output ./MiniMax-H3-Image-Pruned [--source ./MiniMax-H3-Image] [--device cuda]

Without `--source` the needed files of `MiniMaxAI/MiniMax-H3-Image` are downloaded first (about 135 GB). The output is a
Modular Diffusers repository that the stock `MiniMaxH3ImageTransformer2DModel` and `Qwen3VLForConditionalGeneration`
load as they are:

* `transformer/`: `time_embed_dim` 2688 -> 8 and `time_embed_hidden_dim` 5376 -> 16. The new `time_embedder` turns each
  timestep of the sampling grid into a one-hot code (see `timestep_code`), and column k of every `adaln_proj.linear`
  and of `norm_out.linear` holds the original projection's output for grid timestep k, divided by 32. Every other
  tensor is a byte copy.
* `text_encoder/`: the Qwen3-VL conditioner with decoder layers 0-50 and without `lm_head` (tied to the input
  embedding). The pipeline reads `hidden_states[50]`, the output of layer 49; layer 50 stays because the pipeline
  refuses a conditioner with 50 layers or fewer.
* `vae/`, `tokenizer/`, `processor/`, `scheduler/`: unchanged copies.

The original projection outputs are computed on `--device` with the original modules, in the shapes the pipeline
calls them with: one row per text-to-image step, two rows (step, reference) per edit step. The stored value for a step
timestep is the text-to-image one, so text-to-image results are bit-identical to the original's on the build GPU type;
in an edit the step rows can differ from the original's two-row result by single rounding steps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch

from .safetensors_io import NewTensor, read_checkpoint, write_sharded
from .timestep_code import CODE, build_time_embedder, check_codes, code_dims, codes

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
TEXT_ENCODER_KEEP = TEXT_ENCODER_LAYER + 1  # the stock pipeline needs more than 50 decoder layers
NOTICE_LINE = (
    "MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, "
    "Copyright © 2026 MiniMax. All Rights Reserved."
)


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
# AdaLN projections


class _Reader:
    """Tensor access by name across the shards of a checkpoint folder."""

    def __init__(self, folder: Path):
        from safetensors import safe_open

        self._open = safe_open
        self._map = {t.name: t.path for t in read_checkpoint(folder)}
        self._files = {}

    def get(self, name: str) -> torch.Tensor:
        path = self._map[name]
        if path not in self._files:
            self._files[path] = self._open(str(path), framework="pt")
        return self._files[path].get_tensor(name)


def _code_rows(plans, outputs, grid):
    """Row of `outputs` (one tensor per plan, one row per plan timestep) for each grid timestep.

    A step timestep takes its row from the single-timestep (text-to-image) plan. A timestep that occurs only in
    multi-timestep plans (the edit reference timestep) must have the same row in all of them.
    """
    rows = []
    for value in grid:
        single = [out[0] for plan, out in zip(plans, outputs) if plan == [value]]
        shared = [out[plan.index(value)] for plan, out in zip(plans, outputs) if value in plan and len(plan) > 1]
        if single:
            rows.append(single[0])
            continue
        if not all(torch.equal(row, shared[0]) for row in shared):
            raise RuntimeError(f"timestep {value}: the multi-timestep plans disagree on its row")
        rows.append(shared[0])
    return torch.stack(rows)


def _coded_weight(rows: torch.Tensor, columns: int) -> torch.Tensor:
    """`(out_features, columns)` bfloat16 weight whose column k is `rows[k] / CODE`; checked to be exact."""
    weight = torch.zeros(rows.shape[1], columns, dtype=torch.float32)
    weight[:, : rows.shape[0]] = rows.float().T / CODE
    weight = weight.to(torch.bfloat16)
    if not torch.equal((weight[:, : rows.shape[0]].float() * CODE).to(torch.bfloat16).T, rows):
        raise RuntimeError("a projection output is not exactly representable as bfloat16 value / 32")
    return weight


@torch.no_grad()
def build_projections(transformer_dir: Path, plans: list[list[float]], device: str):
    """Original AdaLN projection outputs per plan, and the coded `time_embedder` and projection weights.

    Mirrors `MiniMaxH3ImageTransformer2DModel.forward` loaded in bfloat16: `time_proj` and the float32
    `time_embedder` on the `timestep` tensor of the call, then each bfloat16 projection of `silu(temb)`.
    Returns `(new_tensors, report)`.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageAdaLayerNormModulation

    cfg = json.loads((transformer_dir / "config.json").read_text())
    hidden, temb_dim, freq_dim = cfg["hidden_size"], cfg["time_embed_dim"], cfg["freq_dim"]
    grid = sorted({value for plan in plans for value in plan})
    code_hidden, code_dim = code_dims(len(grid))
    reader = _Reader(transformer_dir)

    def load(module, prefix, dtype):
        module.load_state_dict({k: reader.get(f"{prefix}.{k}") for k in module.state_dict()})
        return module.to(device=device, dtype=dtype).eval()

    time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = load(
        TimestepEmbedding(in_channels=freq_dim, time_embed_dim=cfg["time_embed_hidden_dim"], out_dim=temb_dim),
        "time_embedder",
        torch.float32,
    )
    acts = []
    for plan in plans:
        temb = time_embedder(time_proj(torch.tensor(plan, dtype=torch.float32, device=device)))
        acts.append(torch.nn.functional.silu(temb).to(torch.bfloat16))

    embedder_state = build_time_embedder(grid, freq_dim)
    log(f"time_embedder codes exact: {check_codes(embedder_state, grid, freq_dim, device)}")
    new = {f"time_embedder.{k}": v for k, v in embedder_state.items()}
    coded_embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=code_hidden, out_dim=code_dim)
    coded_embedder.load_state_dict(embedder_state)
    coded_embedder = coded_embedder.to(device=device, dtype=torch.float32).eval()
    plan_codes = [codes(time_proj, coded_embedder, plan, device) for plan in plans]  # what the stock forward feeds

    report = {"grid": grid, "plans": [{"timesteps": plan, "rows": len(plan), "identical_rows": 0, "differing": 0}
                                      for plan in plans]}

    def project(name, original, coded_linear):
        outputs = [original(act).cpu() for act in acts]  # one (len(plan), out_features) tensor per plan
        weight = _coded_weight(_code_rows(plans, outputs, grid), code_dim)
        new[f"{name}.weight"] = weight
        new[f"{name}.bias"] = torch.zeros(weight.shape[0], dtype=torch.bfloat16)
        coded_linear.weight.copy_(weight)
        coded_linear.bias.zero_()
        for entry, out, plan_code in zip(report["plans"], outputs, plan_codes):
            got = coded_linear(plan_code).cpu()
            entry["identical_rows"] += sum(bool(torch.equal(g, o)) for g, o in zip(got, out))
            entry["differing"] += int((got != out).sum())

    modulation = MiniMaxH3ImageAdaLayerNormModulation(time_embed_dim=temb_dim, hidden_size=hidden).to(device)
    coded = torch.nn.Linear(code_dim, modulation.linear.out_features, bias=True, device=device, dtype=torch.bfloat16)
    for i in range(cfg["num_layers"]):
        linear = load(modulation, f"transformer_blocks.{i}.adaln_proj", torch.bfloat16).linear
        project(f"transformer_blocks.{i}.adaln_proj.linear", linear, coded)
        if i % 10 == 9:
            log(f"adaln projections: {i + 1}/{cfg['num_layers']} blocks")
    out_linear = load(torch.nn.Linear(temb_dim, 2 * hidden, bias=True), "norm_out.linear", torch.bfloat16)
    project("norm_out.linear", out_linear, torch.nn.Linear(code_dim, 2 * hidden, device=device, dtype=torch.bfloat16))

    projections = cfg["num_layers"] + 1
    for entry in report["plans"]:
        entry["rows"] *= projections
    return new, {"time_embed_hidden_dim": code_hidden, "time_embed_dim": code_dim, **report}


# ---------------------------------------------------------------------------------------------------------------------
# Components


TEXT_ENCODER_DROP = re.compile(r"^(lm_head\.|model\.language_model\.layers\.(\d+)\.)")


def build_transformer(source: Path, output: Path, plans, device: str, revision: str | None) -> dict:
    src = source / "transformer_diffusers"
    dst = output / "transformer"
    dst.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    new, report = build_projections(src, plans, device)
    log(f"coded projections in {time.perf_counter() - started:.0f} s on {device}")
    for entry in report["plans"]:
        log(f"  plan {entry['timesteps']}: {entry['identical_rows']}/{entry['rows']} projection rows identical to "
            f"the original, {entry['differing']} differing elements")

    stored = read_checkpoint(src)
    names = {t.name for t in stored}
    missing = set(new) - names
    if missing:
        raise RuntimeError(f"replacement tensors without an original: {sorted(missing)}")
    entries = [NewTensor(t.name, new[t.name]) if t.name in new else t for t in stored]
    before = sum(t.nbytes for t in stored)
    after = sum(e.nbytes for e in entries)
    log(f"transformer: {len(stored)} tensors, {len(new)} replaced, {before / 1e9:.2f} -> {after / 1e9:.2f} GB")

    notice = modification_notice(
        f"time_embedder replaced by a one-hot code of the {len(report['grid'])} sampling timesteps; adaln_proj.linear "
        f"and norm_out.linear reduced to {report['time_embed_dim']} input columns holding the original outputs for "
        "those timesteps.",
        revision,
    )
    write_sharded(dst, "diffusion_pytorch_model", entries, {"format": "pt", "notice": notice}, log=log)

    cfg = json.loads((src / "config.json").read_text())
    cfg["time_embed_hidden_dim"] = report["time_embed_hidden_dim"]
    cfg["time_embed_dim"] = report["time_embed_dim"]
    cfg["_timestep_grid"] = report["grid"]
    cfg["_notice"] = notice
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return report


def build_text_encoder(source: Path, output: Path, revision: str | None) -> None:
    src = source / "text_encoder"
    dst = output / "text_encoder"
    dst.mkdir(parents=True, exist_ok=True)

    def keep(name: str) -> bool:
        m = TEXT_ENCODER_DROP.match(name)
        if m is None:
            return True
        return m.group(2) is not None and int(m.group(2)) < TEXT_ENCODER_KEEP

    stored = read_checkpoint(src)
    kept = [t for t in stored if keep(t.name)]
    dropped = sum(t.nbytes for t in stored) - sum(t.nbytes for t in kept)
    log(f"text encoder: {len(stored)} tensors, dropping {len(stored) - len(kept)} ({dropped / 1e9:.2f} GB)")
    notice = modification_notice(
        f"decoder layers {TEXT_ENCODER_KEEP}+ and lm_head removed; the pipeline reads hidden_states"
        f"[{TEXT_ENCODER_LAYER}], the output of layer {TEXT_ENCODER_LAYER - 1}.",
        revision,
    )
    write_sharded(dst, "model", kept, {"format": "pt", "notice": notice}, log=log)

    cfg = json.loads((src / "config.json").read_text())
    text_cfg = cfg["text_config"]
    text_cfg["num_hidden_layers"] = TEXT_ENCODER_KEEP
    for key in ("layer_types",):
        if isinstance(text_cfg.get(key), list):
            text_cfg[key] = text_cfg[key][:TEXT_ENCODER_KEEP]
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
        report = build_transformer(source, output, plans, args.device, args.source_revision)
        (output / "transformer" / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
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
