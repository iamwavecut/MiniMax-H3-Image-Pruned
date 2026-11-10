"""Generate or edit one image with MiniMax-H3-Image-Pruned, using only Diffusers and Transformers.

    python generate.py --prompt "The image is ..." --size 4736x3552 --out image.png
    python generate.py --prompt-file prompt.txt --image reference.png --size 4096x4096 --out edit.png

The model expects a long, complete English description of the whole image; an edit prompt names what to keep, then
what to change, and refers to the references as <Picture 1>, <Picture 2>, ...
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--prompt")
ap.add_argument("--prompt-file", type=Path)
ap.add_argument("--image", action="append", default=[], help="reference image (repeat for <Picture 2>, ...)")
ap.add_argument("--size", default="2048x2048", help="WIDTHxHEIGHT of the returned image")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--repo", default="WaveCut/MiniMax-H3-Image-Pruned", help="Hub repository or local folder")
ap.add_argument("--resident", action="store_true", help="keep every component on the GPU (about 95 GB)")
ap.add_argument("--out", default="image.png")
args = ap.parse_args()

manager = ComponentsManager()
if not args.resident:
    manager.enable_auto_cpu_offload(device="cuda")
pipe = ModularPipeline.from_pretrained(args.repo, components_manager=manager)
load_kwargs = {"dtype": torch.bfloat16}
if Path(args.repo).is_dir():  # a local copy: its modular_model_index.json still names the Hub repository
    load_kwargs["pretrained_model_name_or_path"] = args.repo
pipe.load_components(**load_kwargs)
if args.resident:
    pipe.to("cuda")

prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text().strip()
width, height = (int(v) for v in args.size.lower().split("x"))
kwargs = dict(prompt=prompt, height=height, width=width, generator=torch.Generator().manual_seed(args.seed),
              output="images")
if args.image:
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in args.image]
    kwargs["image"] = images if len(images) > 1 else images[0]
started = time.perf_counter()
image = pipe(**kwargs)[0]
image.save(args.out)
print(f"{args.out}: {image.width}x{image.height} in {time.perf_counter() - started:.1f} s")
