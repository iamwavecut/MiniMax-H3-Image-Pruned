"""A `time_embedder` that maps each timestep of the sampling grid to a one-hot code.

The stock `MiniMaxH3ImageTransformer2DModel` computes `temb = time_embedder(time_proj(t))` in float32, and every
AdaLN projection (`adaln_proj.linear`, `norm_out.linear`) reads `silu(temb).to(bfloat16)`. The weights built here
give, for grid timestep t_k:

    temb[k] = 32, temb[j] < -300 for every j != k,

so `silu(temb).to(bfloat16)` is exactly `32 * e_k`: `silu(32)` is 32 in float32 and in bfloat16, and `silu(x)` is
0 for x below about -89 (`exp(-x)` overflows float32). A projection whose column k holds `row_k / 32` then returns
`row_k` exactly: the matrix product has a single nonzero term, and scaling a bfloat16 value by a power of two is
exact. The result does not depend on the GPU, on the number of rows in the call, or on the GEMM algorithm.

Construction. `time_proj` feature `freq_dim // 2` is `u = sin(t)`, increasing on the grid (t in [0, 1]). For each
gap between neighbouring grid points, with midpoint m, the hidden layer has two hinge units, `silu(A (u - m))`
and `silu(-A (u - m))`. At every grid point their pre-activation is at least `A * gap / 2 >= HINGE` in magnitude,
so each unit is either 0 or larger than HINGE. Output k starts at 32 (its bias) and subtracts `GAIN` times the
hinge above its upper midpoint and the hinge below its lower midpoint. At t_k both hinges are 0. At any other grid
point one of them exceeds HINGE, so the output is below 32 - GAIN * HINGE.

The margins absorb float32 rounding with a wide reserve; the codes stay exact also with TF32 matmuls and with the
`time_embedder` cast to bfloat16 (`check_codes` tests all three). A timestep off the grid gives no valid code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

CODE = 32.0  # value of the active code entry; a power of two, so `row / CODE` is exact in bfloat16
HINGE = 128.0  # minimum |pre-activation| of a hinge unit at a grid point
GAIN = 4.0  # an inactive output is at most CODE - GAIN * HINGE = -480
OFF = -1024.0  # bias of padding units: silu(OFF) == 0


def _pad8(n: int) -> int:
    return max(8, -(-n // 8) * 8)


def code_dims(num_timesteps: int) -> tuple[int, int]:
    """`(time_embed_hidden_dim, time_embed_dim)` for a grid of `num_timesteps` values."""
    return _pad8(2 * (num_timesteps - 1)), _pad8(num_timesteps)


def build_time_embedder(timesteps: Sequence[float], freq_dim: int) -> dict[str, torch.Tensor]:
    """float32 state dict of the stock `TimestepEmbedding(freq_dim, hidden, out)` for the sorted grid `timesteps`."""
    ts = [float(t) for t in timesteps]
    if ts != sorted(ts) or len(set(ts)) != len(ts):
        raise ValueError(f"timesteps must be sorted and distinct, got {ts}")
    if not all(0.0 <= t <= 1.0 for t in ts):
        raise ValueError(f"timesteps must lie in [0, 1] (sin is increasing there), got {ts}")
    n = len(ts)
    hidden, out = code_dims(n)
    sin_index = freq_dim // 2
    # float32 values of the timesteps, as the pipeline passes them, then sin in float64 for the design
    u = [math.sin(float(torch.tensor(t, dtype=torch.float32))) for t in ts]
    gaps = [b - a for a, b in zip(u, u[1:])]
    slope = 2.0 ** math.ceil(math.log2(2 * HINGE / min(gaps)))
    mids = [(a + b) / 2 for a, b in zip(u, u[1:])]

    w1 = torch.zeros(hidden, freq_dim, dtype=torch.float64)
    b1 = torch.full((hidden,), OFF, dtype=torch.float64)
    w2 = torch.zeros(out, hidden, dtype=torch.float64)
    b2 = torch.full((out,), OFF, dtype=torch.float64)
    for g, m in enumerate(mids):
        up, down = 2 * g, 2 * g + 1  # up: positive above m; down: positive below m
        w1[up, sin_index], b1[up] = slope, -slope * m
        w1[down, sin_index], b1[down] = -slope, slope * m
    for k in range(n):
        b2[k] = CODE
        if k + 1 < n:
            w2[k, 2 * k] = -GAIN  # hinge above the midpoint between t_k and t_{k+1}
        if k > 0:
            w2[k, 2 * (k - 1) + 1] = -GAIN  # hinge below the midpoint between t_{k-1} and t_k
    state = {"linear_1.weight": w1, "linear_1.bias": b1, "linear_2.weight": w2, "linear_2.bias": b2}
    return {name: value.to(torch.float32) for name, value in state.items()}


@torch.no_grad()
def codes(time_proj, time_embedder, timesteps: Sequence[float], device, dtype=torch.bfloat16) -> torch.Tensor:
    """`silu(time_embedder(time_proj(t))).to(dtype)` for each timestep, as the stock transformer computes it."""
    param = next(time_embedder.parameters())
    t = torch.tensor(list(timesteps), dtype=torch.float32, device=device)
    temb = time_embedder(time_proj(t).to(param.dtype))
    return torch.nn.functional.silu(temb).to(dtype)


@torch.no_grad()
def check_codes(state: dict[str, torch.Tensor], timesteps: Sequence[float], freq_dim: int, device) -> dict:
    """Build the stock modules with `state` and check that every grid timestep gives exactly `CODE * e_k`.

    Checked three ways: float32 matmuls, TF32 matmuls (when on CUDA), and the `time_embedder` cast to bfloat16.
    Raises `AssertionError` on a mismatch; returns the checked variants.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    hidden, out = state["linear_1.weight"].shape[0], state["linear_2.weight"].shape[0]
    time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    probe = time_proj(torch.tensor(list(timesteps), dtype=torch.float32))[:, freq_dim // 2]
    torch.testing.assert_close(probe, torch.sin(torch.tensor(list(timesteps), dtype=torch.float32)))

    expected = torch.zeros(len(timesteps), out, dtype=torch.bfloat16, device=device)
    expected[torch.arange(len(timesteps)), torch.arange(len(timesteps))] = CODE
    variants = {}
    for name, dtype, tf32 in (("float32", torch.float32, False), ("tf32", torch.float32, True),
                              ("bfloat16", torch.bfloat16, False)):
        if tf32 and torch.device(device).type != "cuda":
            continue
        embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=hidden, out_dim=out)
        embedder.load_state_dict(state)
        embedder = embedder.to(device=device, dtype=dtype).eval()
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            got = codes(time_proj, embedder, timesteps, device)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        if not torch.equal(got, expected):
            raise AssertionError(f"time_embedder codes are not one-hot ({name}):\n{got.float().cpu()}")
        variants[name] = True
    return variants
