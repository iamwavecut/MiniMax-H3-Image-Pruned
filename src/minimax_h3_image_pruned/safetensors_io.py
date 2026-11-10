"""Byte-exact safetensors filtering: tensors are copied as stored, without a round trip through torch."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
CHUNK = 64 << 20


@dataclass
class StoredTensor:
    """A tensor of an existing safetensors file, addressed by absolute byte offsets."""

    path: Path
    name: str
    dtype: str
    shape: list[int]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class NewTensor:
    name: str
    tensor: torch.Tensor

    @property
    def dtype(self) -> str:
        return DTYPE_NAMES[self.tensor.dtype]

    @property
    def shape(self) -> list[int]:
        return list(self.tensor.shape)

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def read_tensors(path: Path) -> list[StoredTensor]:
    """The tensors of one file, in storage order."""
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    base = 8 + header_size
    tensors = [
        StoredTensor(Path(path), name, info["dtype"], info["shape"], base + info["data_offsets"][0],
                     base + info["data_offsets"][1])
        for name, info in header.items()
    ]
    return sorted(tensors, key=lambda t: t.start)


def read_checkpoint(folder: Path) -> list[StoredTensor]:
    """Every tensor of a (possibly sharded) checkpoint folder, in shard and storage order."""
    folder = Path(folder)
    indexes = sorted(folder.glob("*.safetensors.index.json"))
    if indexes:
        shards = sorted(set(json.loads(indexes[0].read_text())["weight_map"].values()))
    else:
        shards = sorted(p.name for p in folder.glob("*.safetensors"))
    tensors = []
    for shard in shards:
        tensors.extend(read_tensors(folder / shard))
    return tensors


def _header(entries, metadata) -> bytes:
    header, offset = {}, 0
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for entry in entries:
        header[entry.name] = {"dtype": entry.dtype, "shape": entry.shape, "data_offsets": [offset, offset + entry.nbytes]}
        offset += entry.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    return raw + b" " * (-len(raw) % 8)


def write_file(path: Path, entries, metadata: dict | None = None) -> int:
    """Write `entries` (StoredTensor or NewTensor) to one safetensors file. Returns the bytes written."""
    raw = _header(entries, metadata)
    tmp = Path(str(path) + ".partial")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        handles = {}
        try:
            for entry in entries:
                if isinstance(entry, NewTensor):
                    data = entry.tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    out.write(data)
                    continue
                src = handles.get(entry.path) or handles.setdefault(entry.path, open(entry.path, "rb"))
                src.seek(entry.start)
                left = entry.nbytes
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        raise IOError(f"{entry.path} ended inside {entry.name}")
                    out.write(block)
                    left -= len(block)
        finally:
            for handle in handles.values():
                handle.close()
    os.replace(tmp, path)
    return 8 + len(raw) + sum(e.nbytes for e in entries)


def write_sharded(folder: Path, prefix: str, entries, metadata: dict | None = None,
                  max_shard_bytes: int = 5_000_000_000, log=print) -> dict:
    """Write `entries` as `<prefix>-0000i-of-0000N.safetensors` plus `<prefix>.safetensors.index.json`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shards, current, size = [], [], 0
    for entry in entries:
        if current and size + entry.nbytes > max_shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry.nbytes
    if current:
        shards.append(current)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        name = f"{prefix}-{i:05d}-of-{len(shards):05d}.safetensors"
        written = write_file(folder / name, shard, metadata)
        total += sum(e.nbytes for e in shard)
        weight_map.update({e.name: name for e in shard})
        log(f"wrote {folder.name}/{name}: {len(shard)} tensors, {written / 1e9:.2f} GB")
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (folder / f"{prefix}.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
