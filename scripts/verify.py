"""Check the pruned MiniMax-H3-Image against the original release, and time both.

    python verify.py reference --source ./MiniMax-H3-Image --work ./verify
    python verify.py pruned --pruned ./MiniMax-H3-Image-Pruned --work ./verify [--resident]
    python verify.py compare --work ./verify

`reference` runs the official Modular Diffusers pipeline (bfloat16, auto CPU offload) on the cases below and records
every transformer call. `pruned` runs the pruned pipeline on the same cases and also feeds the recorded calls to the
pruned transformer, with and without `skip_dead_rows`. `compare` reports, per case: whether the text conditioning, the
transformer outputs and the images are bit-identical, and the time and GPU memory of each run.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

T2I_PROMPT = "examples/revised_prompt.t2i.txt"
EDIT_PROMPT = "examples/revised_prompt.edit.txt"
SHORT_PROMPT = (
    "A rustic, overhead shot of a farm-to-table feast, showcasing a variety of colorful, seasonal vegetables, "
    "artisanal cheeses, and freshly baked bread on a weathered wooden table."
)
CASES = [
    {"id": "t2i_a", "prompt_file": T2I_PROMPT, "size": "2048x2048", "seed": 42, "capture": True},
    {"id": "t2i_b", "prompt": SHORT_PROMPT, "size": "2048x2048", "seed": 7, "capture": True},
    {"id": "edit_a", "prompt_file": EDIT_PROMPT, "image": "examples/reference1.png", "size": "2048x2048", "seed": 1234,
     "capture": True},
    {"id": "t2i_full", "prompt_file": T2I_PROMPT, "size": "4736x3552", "seed": 42, "capture": False},
]


def log(**kw):
    print(json.dumps({"ts": time.strftime("%H:%M:%S"), **kw}), flush=True)


class Peak:
    """Whole-GPU memory peak from NVML, sampled every 5 ms."""

    def __init__(self, index=0):
        import pynvml

        pynvml.nvmlInit()
        self._nvml, self._handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
        self.peak, self._stop = 0, threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            time.sleep(0.005)

    def __enter__(self):
        self.peak = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def case_inputs(case, source: Path):
    prompt = case.get("prompt") or (source / case["prompt_file"]).read_text().strip()
    width, height = (int(v) for v in case["size"].split("x"))
    kwargs = dict(prompt=prompt, height=height, width=width, generator=torch.Generator().manual_seed(case["seed"]),
                  output="images")
    if "image" in case:
        kwargs["image"] = Image.open(source / case["image"]).convert("RGB")
    return kwargs


def cpu(value):
    return value.detach().to("cpu") if torch.is_tensor(value) else value


def instrument(pipe, state):
    """Hooks: per-stage time and, when `state['capture']`, every transformer call (inputs and output)."""

    def dit_pre(module, args, kwargs):
        torch.cuda.synchronize()
        state["t_dit"] = time.perf_counter()

    def dit_post(module, args, kwargs, output):
        torch.cuda.synchronize()
        state["dit_s"] += time.perf_counter() - state["t_dit"]
        if state["capture"]:
            out = output[0] if isinstance(output, tuple) else output.sample
            state["calls"].append({"kwargs": {k: cpu(v) for k, v in kwargs.items()}, "output": cpu(out)})

    def te_pre(module, args, kwargs):
        torch.cuda.synchronize()
        state["t_te"] = time.perf_counter()

    def te_post(module, args, kwargs, output):
        torch.cuda.synchronize()
        state["te_s"] += time.perf_counter() - state["t_te"]

    pipe.transformer.register_forward_pre_hook(dit_pre, with_kwargs=True)
    pipe.transformer.register_forward_hook(dit_post, with_kwargs=True)
    pipe.text_encoder.model.register_forward_pre_hook(te_pre, with_kwargs=True)
    pipe.text_encoder.model.register_forward_hook(te_post, with_kwargs=True)


def run_cases(pipe, source: Path, out_dir: Path, cases, label: str, warmup: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"capture": False, "calls": [], "te_s": 0.0, "dit_s": 0.0}
    instrument(pipe, state)
    records = []
    if warmup:  # the first call compiles kernels and fills the allocator; it is not measured
        state["capture"] = False
        pipe(**case_inputs(cases[0], source))
    for case in cases:
        state.update(capture=case["capture"], calls=[], te_s=0.0, dit_s=0.0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with Peak() as peak:
            started = time.perf_counter()
            image = pipe(**case_inputs(case, source))[0]
            torch.cuda.synchronize()
            total = time.perf_counter() - started
        image.save(out_dir / f"{case['id']}.png")
        if case["capture"]:
            torch.save(state["calls"], out_dir / f"{case['id']}.calls.pt")
        rec = {"run": label, "id": case["id"], "size": case["size"], "total_s": round(total, 2),
               "te_s": round(state["te_s"], 2), "dit_s": round(state["dit_s"], 2),
               "nvml_peak_gib": round(peak.peak / 2**30, 2),
               "torch_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        records.append(rec)
        log(**rec)
    with (out_dir / "records.jsonl").open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def reference(args):
    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(str(args.source), components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(args.source))
    for name in ("transformer", "text_encoder", "vae", "tokenizer", "processor", "scheduler"):
        if getattr(pipe, name, None) is None:
            raise RuntimeError(f"{name} did not load")
    log(loaded="reference", te_layers=pipe.text_encoder.config.text_config.num_hidden_layers)
    run_cases(pipe, args.source, args.work / "reference", CASES, "reference")
    print("REFERENCE_DONE", flush=True)


@torch.no_grad()
def transformer_check(transformer, ref_dir: Path):
    """Feed the recorded reference calls to the pruned transformer, with and without `skip_dead_rows`."""
    rows = []
    for calls_file in sorted(ref_dir.glob("*.calls.pt")):
        for call in torch.load(calls_file, weights_only=False):
            kwargs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in call["kwargs"].items()}
            expected = call["output"].float()
            for skip in (False, True):
                transformer.skip_dead_rows = skip
                got = transformer(**kwargs)
                got = (got[0] if isinstance(got, tuple) else got.sample).float().cpu()
                timestep_indices = kwargs["timestep_indices"].cpu()
                image_indices = kwargs["image_indices"].cpu()
                n_ref = int((timestep_indices[image_indices] == kwargs["timestep"].numel() - 1).sum()) \
                    if kwargs["timestep"].numel() > 1 else 0
                live = slice(n_ref if skip else 0, None)  # reference rows are zero with skip_dead_rows
                diff = (got[:, live] - expected[:, live]).abs()
                rows.append({"case": calls_file.name.split(".")[0], "step": int(kwargs["pdd_step_index"]),
                             "skip_dead_rows": skip, "identical": bool(torch.equal(got[:, live], expected[:, live])),
                             "max_abs": float(diff.max()),
                             "rel": float(diff.norm() / expected[:, live].norm())})
    transformer.skip_dead_rows = True
    return rows


def pruned(args):
    from minimax_h3_image_pruned import load_pipeline

    pipe = load_pipeline(str(args.pruned), offload=not args.resident)
    label = "pruned-resident" if args.resident else "pruned"
    log(loaded=label, te_layers=len(pipe.text_encoder.model.language_model.layers))
    cases = CASES if not args.resident else [c | {"capture": False} for c in CASES]
    run_cases(pipe, args.source or args.pruned, args.work / label, cases, label)
    if args.resident:
        print("PRUNED_DONE", flush=True)
        return
    rows = transformer_check(pipe.transformer, args.work / "reference")
    (args.work / "transformer_check.json").write_text(json.dumps(rows, indent=1))
    log(transformer_check={"calls": len(rows), "identical": sum(r["identical"] for r in rows)})
    # The full last block end to end: the images should match the reference bit for bit.
    pipe.transformer.skip_dead_rows = False
    full_cases = [c | {"id": c["id"] + "_full", "capture": False} for c in CASES if c["capture"]]
    run_cases(pipe, args.source or args.pruned, args.work / label, full_cases, label + "-full", warmup=False)
    print("PRUNED_DONE", flush=True)


def image_diff(a: Path, b: Path):
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    if x.shape != y.shape:
        return {"identical": False, "shape": [list(x.shape), list(y.shape)]}
    d = np.abs(x - y)
    mse = float((d.astype(np.float64) ** 2).mean())
    return {"identical": bool(mse == 0), "max_abs": int(d.max()), "differing_pixels": int((d.max(-1) > 0).sum()),
            "psnr_db": None if mse == 0 else round(10 * np.log10(255**2 / mse), 2)}


def conditioning_diff(a: Path, b: Path):
    x, y = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    ex, ey = x[0]["kwargs"]["encoder_hidden_states"], y[0]["kwargs"]["encoder_hidden_states"]
    return {"identical": bool(torch.equal(ex, ey)), "shape": list(ex.shape),
            "max_abs": float((ex.float() - ey.float()).abs().max()) if ex.shape == ey.shape else None}


def compare(args):
    ref = args.work / "reference"
    result = {"images": {}, "conditioning": {}, "runs": []}
    for case in CASES:
        for label, suffix in (("pruned", ""), ("pruned", "_full"), ("pruned-resident", "")):
            got = args.work / label / f"{case['id']}{suffix}.png"
            if got.exists() and (ref / f"{case['id']}.png").exists():
                result["images"][f"{label}/{case['id']}{suffix}"] = image_diff(ref / f"{case['id']}.png", got)
        calls_ref, calls_got = ref / f"{case['id']}.calls.pt", args.work / "pruned" / f"{case['id']}.calls.pt"
        if calls_ref.exists() and calls_got.exists():
            result["conditioning"][case["id"]] = conditioning_diff(calls_ref, calls_got)
    for records in sorted(args.work.glob("*/records.jsonl")):
        result["runs"].extend(json.loads(line) for line in records.read_text().splitlines())
    check = args.work / "transformer_check.json"
    if check.exists():
        rows = json.loads(check.read_text())
        for skip in (False, True):
            sel = [r for r in rows if r["skip_dead_rows"] == skip]
            result[f"transformer_skip_{str(skip).lower()}"] = {
                "calls": len(sel), "identical": sum(r["identical"] for r in sel),
                "max_abs": max(r["max_abs"] for r in sel), "max_rel": max(r["rel"] for r in sel)}
    (args.work / "compare.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["reference", "pruned", "compare"])
    ap.add_argument("--source", type=Path, help="local MiniMaxAI/MiniMax-H3-Image (prompts, reference image)")
    ap.add_argument("--pruned", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--resident", action="store_true", help="pruned: keep every component on the GPU (no offload)")
    args = ap.parse_args()
    {"reference": reference, "pruned": pruned, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
