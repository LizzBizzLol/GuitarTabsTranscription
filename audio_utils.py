"""Amplitude log-Mel features and crash-safe artifact writes."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 256
    n_mels: int = 256

    def __post_init__(self):
        for value in asdict(self).values():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Audio parameters must be positive integers")
        if self.n_fft % 2:
            raise ValueError("n_fft must be even for the documented centered grid")


def external_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    root = Path(__file__).resolve().parent
    if path == root or root in path.parents:
        raise ValueError(f"Store data, caches and checkpoints outside the repository: {path}")
    return path


def _atomic_write(path: str | Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(value, path: str | Path) -> None:
    _atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_json_save(value, path: str | Path) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(payload))


def compute_log_mel(
    waveform: torch.Tensor,
    input_sample_rate: int,
    *,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 256,
    n_mels: int = 256,
    source: str = "<waveform>",
) -> torch.Tensor:
    """Return [time, mel] float32 features; frame t is at t*hop/sample_rate."""
    cfg = AudioConfig(sample_rate, n_fft, hop_length, n_mels)
    if not isinstance(input_sample_rate, int) or input_sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for {source}")
    if waveform.ndim not in (1, 2) or waveform.numel() == 0:
        raise ValueError(f"Expected nonempty [samples] or [channels, samples] audio: {source}")
    waveform = waveform.to(dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite audio samples: {source}")
    mono = waveform.mean(dim=0) if waveform.ndim == 2 else waveform
    if input_sample_rate != cfg.sample_rate:
        mono = torchaudio.functional.resample(mono, input_sample_rate, cfg.sample_rate)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels, f_min=0.0,
        f_max=cfg.sample_rate / 2, power=1.0, center=True,
        pad_mode="constant", window_fn=torch.hann_window, normalized=False,
        norm=None, mel_scale="htk",
    ).to(device=mono.device, dtype=mono.dtype)
    result = torch.log10(transform(mono) + 1e-5).transpose(0, 1).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"Non-finite log-Mel features: {source}")
    return result


def load_log_mel(path: str | Path, config: AudioConfig = AudioConfig()) -> torch.Tensor:
    path = Path(path)
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot decode audio {path}: {exc}") from exc
    with torch.no_grad():
        return compute_log_mel(waveform, sample_rate, source=str(path), **asdict(config))


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
