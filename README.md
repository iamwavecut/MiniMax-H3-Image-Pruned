# MiniMax-H3-Image-Pruned

Build and load scripts for [WaveCut/MiniMax-H3-Image-Pruned](https://huggingface.co/WaveCut/MiniMax-H3-Image-Pruned),
a smaller checkpoint of [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) that produces
the same images. The scripts work with Diffusers (Modular Diffusers pipeline) and Transformers (the Qwen3-VL text
encoder).

![Generation time, GPU memory and checkpoint size](assets/speed_memory_size.png)

| | Original | Pruned |
|---|---|---|
| Transformer | 66.3 GB | 40.3 GB |
| Text encoder | 66.7 GB | 51.5 GB |
| VAE | 2.3 GB | 2.3 GB |
| Time per 4736×3552 image, RTX PRO 6000 (96 GB) | 101.2 s (model offload) | 26.3 s |
| Images | | bit-identical |

The time difference comes from memory: the original does not fit a 96 GB card and the offload manager moves the
transformer and the text encoder for every image; the pruned models fit together. Per-step compute is unchanged.

## What is removed

| Component | Removed | Why it is not needed |
|---|---|---|
| Transformer | `transformer_blocks.*.adaln_proj.linear` (13.0B parameters), `time_embedder`, `norm_out.linear` | Their input is the timestep alone, and the checkpoint samples on one fixed grid: six steps plus t = 0.999 for the reference rows of an edit. Their outputs for that grid are stored as tables (116 MB). The audio third of every projection, which the image model never addresses, is not stored at all. |
| Transformer, last block | Queries, attention output and feed-forward of the text rows and reference rows | Nothing reads these rows after the last block; they still provide keys and values. Runtime only, `transformer.skip_dead_rows`. |
| Text encoder | Decoder layers 50-63 and `lm_head` (15.2B parameters) | The transformer is conditioned on `hidden_states[50]`, the output of layer 49 before the final norm. |

The VAE, tokenizer, processor and scheduler are the original files.

### Modulation tables

Each block's AdaLN projection maps `silu(time_embedder(t))` to six modulation vectors for each of three modalities.
`prune.py` runs the original `time_proj`, `time_embedder` and projection modules on every `timestep` tensor the pipeline
passes (one value per text-to-image step; the step value and 0.999 per edit step) and keeps the image and text rows of
the result. The pruned model looks the rows up instead of computing them, then runs the original block code.

Matrix products can round differently with a different number of rows, so the tables keep each timestep set as the
pipeline sends it: one row for a text-to-image step, two rows for an edit step. Built on the same GPU type, the pruned
transformer returns bit-identical outputs. On another GPU type the original model's own adaptive-norm outputs differ by
single bfloat16 rounding steps in a few entries; the tables keep the values of the build GPU.

The tables cover only the release's sampling grid. Any other timestep raises an error.

## Use

```bash
pip install "git+https://github.com/iamwavecut/MiniMax-H3-Image-Pruned"
```

The package needs Diffusers with MiniMax-H3-Image support (merged after 0.41; until a release, install
`diffusers @ git+https://github.com/huggingface/diffusers@875373ac7f5856071e345a3e394c74efccc6b9e6`) and
`transformers>=5.22`.

```python
import torch
from minimax_h3_image_pruned import load_pipeline

pipe = load_pipeline("WaveCut/MiniMax-H3-Image-Pruned")  # auto CPU offload; offload=False keeps all on the GPU
image = pipe(
    prompt=open("prompt.txt").read(),
    height=3552,
    width=4736,
    generator=torch.Generator().manual_seed(42),
    output="images",
)[0]
```

Edit: pass `image=reference` (or a list) and refer to the references as `<Picture 1>`, `<Picture 2>`, ... in the
prompt. `scripts/generate.py` wraps both.

### Components separately

Transformer, with Diffusers' remote code instead of this package:

```python
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "WaveCut/MiniMax-H3-Image-Pruned", subfolder="transformer", trust_remote_code=True, torch_dtype=torch.bfloat16
)
```

Text encoder (Transformers). The checkpoint holds layers 0-49. `adapt_text_encoder` replaces the final norm with an
identity, so the last hidden state is the conditioning, and reports 51 layers to pass the pipeline's depth check:

```python
from transformers import Qwen3VLForConditionalGeneration
from minimax_h3_image_pruned import adapt_text_encoder

text_encoder = adapt_text_encoder(
    Qwen3VLForConditionalGeneration.from_pretrained(
        "WaveCut/MiniMax-H3-Image-Pruned", subfolder="text_encoder", dtype=torch.bfloat16
    )
)
hidden = text_encoder.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[50]
```

`lm_head` is tied to the input embedding so that loading allocates nothing for it; the truncated model is not a usable
language model.

## Build the checkpoint yourself

```bash
minimax-h3-image-prune --output ./MiniMax-H3-Image-Pruned --device cuda
```

It downloads the needed files of `MiniMaxAI/MiniMax-H3-Image` (135 GB) unless `--source` names a local copy, copies
every kept tensor byte for byte, and computes the tables on `--device` (5 s and about 1 GB of GPU memory on an RTX
PRO 6000; the whole build took 59 s on a pod with fast local disk). `--skip text_encoder copies` builds only the
transformer.

## Verify

```bash
python scripts/verify.py reference --source ./MiniMax-H3-Image --work ./verify
python scripts/verify.py pruned --source ./MiniMax-H3-Image --pruned ./MiniMax-H3-Image-Pruned --work ./verify
python scripts/verify.py compare --work ./verify
```

`reference` runs the official pipeline and records every transformer call; `pruned` runs the pruned pipeline on the
same prompts and seeds and replays the recorded calls through the pruned transformer. `compare` reports whether the
text conditioning, transformer outputs and images are bit-identical, with time and GPU memory for each run.

## Limits

* Only the release's six-step grid. The pruned transformer cannot run other timesteps or step counts.
* Diffusers context parallelism and tensor parallelism plans are not carried over.
* SGLang and ComfyUI load the original layout; they need their own loaders for this checkpoint.

## License

The scripts are Apache-2.0. The model weights are MiniMax-H3-Image Works under the
[MiniMax-H3-Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE): use and
distribution are limited to the Applicable Territory, which excludes the European Union, the United Kingdom, the
Republic of Korea and the United States of America.
