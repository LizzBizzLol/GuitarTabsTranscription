"""SynthTab JAMS conversion, stable composition splits and frame datasets."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio_utils import (
    AudioConfig, atomic_json_save, atomic_torch_save, external_path,
    file_identity, load_log_mel,
)

STRING_ORDER = (6, 5, 4, 3, 2, 1)
SILENCE = 25
VERSION = 1


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrackRecord:
    audio_path: str
    labels_path: str
    group_id: str

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("A nonempty source-composition group_id is required")


def split_for_group(group_id: str, seed: int = 42) -> str:
    value = int(_digest([seed, group_id]), 16) / (1 << 256)
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def split_tracks(records: list[TrackRecord], seed: int = 42) -> dict[str, list[TrackRecord]]:
    result = {name: [] for name in ("train", "val", "test")}
    for record in records:
        result[split_for_group(record.group_id, seed)].append(record)
    return result


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def jams_to_labels(jams_path: str | Path, num_frames: int,
                   config: AudioConfig = AudioConfig()) -> tuple[torch.Tensor, list[int]]:
    """Read SynthTab tick-based JAMS and rasterize half-open note intervals."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    try:
        jam = json.loads(Path(jams_path).read_text(encoding="utf-8-sig"))
        annotations = jam["annotations"]
        tempo_annotations = [a for a in annotations if a["namespace"] == "tempo"]
        if len(tempo_annotations) != 1:
            raise ValueError("Exactly one tempo annotation is required")
        tempos = sorted(tempo_annotations[0]["data"], key=lambda event: event["time"])
        if not tempos:
            raise ValueError("Empty tempo map")
        segments = []
        expected = elapsed = 0.0
        for event in tempos:
            onset = _number(event["time"], "tempo onset")
            duration = _number(event["duration"], "tempo duration")
            bpm = _number(event["value"], "tempo BPM")
            if abs(onset - expected) > 1e-7 or duration <= 0 or bpm <= 0:
                raise ValueError("Tempo map must cover ticks continuously from zero with positive BPM")
            seconds_per_tick = 60.0 / (bpm * 960)
            segments.append((onset, onset + duration, elapsed, seconds_per_tick))
            elapsed += duration * seconds_per_tick
            expected = onset + duration

        def seconds(tick):
            tick = _number(tick, "note tick")
            if tick < 0 or tick > expected + 1e-7:
                raise ValueError(f"Note tick {tick} outside tempo coverage [0, {expected}]")
            for start, stop, total, scale in segments:
                if tick <= stop:
                    return total + (tick - start) * scale
            return elapsed

        strings = {}
        for annotation in annotations:
            if annotation["namespace"] != "note_tab":
                continue
            sandbox = annotation["sandbox"]
            index = sandbox["string_index"]
            tuning = sandbox["open_tuning"]
            if type(index) is not int or index not in STRING_ORDER or index in strings:
                raise ValueError(f"Invalid/duplicate string_index: {index}")
            if type(tuning) is not int or not 0 <= tuning <= 127:
                raise ValueError(f"Invalid tuning for string {index}")
            notes = []
            for note in annotation["data"]:
                fret = note["value"]["fret"]
                onset = _number(note["time"], "note onset")
                duration = _number(note["duration"], "note duration")
                if type(fret) is not int or not 0 <= fret <= 24 or duration <= 0:
                    raise ValueError(f"Unsupported fret/duration on string {index}: {fret}, {duration}")
                notes.append((seconds(onset), seconds(onset + duration), fret))
            # Merge repeated same-fret intervals; conflicting overlaps cannot be represented.
            merged = []
            for start, stop, fret in sorted(notes):
                if merged and start < merged[-1][1] - 1e-10:
                    if fret != merged[-1][2]:
                        raise ValueError(f"Conflicting simultaneous frets on string {index}")
                    merged[-1] = (merged[-1][0], max(stop, merged[-1][1]), fret)
                else:
                    merged.append((start, stop, fret))
            strings[index] = (tuning, merged)
        if set(strings) != set(STRING_ORDER):
            raise ValueError("Exactly six distinct string annotations are required")
        labels = torch.full((num_frames, 6), SILENCE, dtype=torch.long)
        def frame_bound(time):
            grid = time * config.sample_rate / config.hop_length
            if abs(grid - round(grid)) < 1e-9:
                grid = round(grid)
            return max(0, min(num_frames, math.ceil(grid)))

        for column, index in enumerate(STRING_ORDER):
            for start, stop, fret in strings[index][1]:
                labels[frame_bound(start):frame_bound(stop), column] = fret
        return labels, [strings[index][0] for index in STRING_ORDER]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SynthTab annotation {jams_path}: {exc}") from exc


def feature_cache_path(audio_path: str | Path, cache_dir: str | Path,
                       config: AudioConfig) -> Path:
    key = _digest({"version": VERSION, "audio": file_identity(audio_path), "config": asdict(config)})
    return external_path(cache_dir) / (key + ".pt")


def cached_features(audio_path: str | Path, cache_dir: str | Path,
                    config: AudioConfig = AudioConfig()) -> torch.Tensor:
    path = feature_cache_path(audio_path, cache_dir, config)
    if not path.exists():
        atomic_torch_save(load_log_mel(audio_path, config).cpu(), path)
    try:
        features = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != config.n_mels or features.dtype != torch.float32
                or features.shape[0] == 0 or not torch.isfinite(features).all()):
            raise ValueError("Invalid feature tensor")
        return features
    except Exception as exc:
        raise ValueError(f"Invalid feature cache {path}; remove it to regenerate: {exc}") from exc


def prepare_track(audio_path: str | Path, jams_path: str | Path, group_id: str,
                  labels_path: str | Path, cache_dir: str | Path,
                  config: AudioConfig = AudioConfig()) -> TrackRecord:
    labels_path = external_path(labels_path)
    features = cached_features(audio_path, cache_dir, config)
    labels, tuning = jams_to_labels(jams_path, len(features), config)
    payload = {
        "version": VERSION, "labels": labels, "audio_config": asdict(config),
        "center": True, "string_order": list(STRING_ORDER), "silence_class": SILENCE,
        "tuning": tuning, "group_id": group_id, "audio": file_identity(audio_path),
        "annotation": file_identity(jams_path),
    }
    record = TrackRecord(str(Path(audio_path).resolve()), str(labels_path), group_id)
    atomic_torch_save(payload, labels_path)
    return record


class GuitarDataset(Dataset):
    def __init__(self, records: list[TrackRecord | dict], cache_dir: str | Path, *,
                 context_frames: int = 1, config: AudioConfig = AudioConfig(),
                 split: str | None = None, seed: int = 42, lru_size: int = 8):
        if type(context_frames) is not int or context_frames <= 0 or type(lru_size) is not int or lru_size <= 0:
            raise ValueError("context_frames and lru_size must be positive integers")
        if split not in (None, "train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.records = [r if isinstance(r, TrackRecord) else TrackRecord(**r) for r in records]
        if split is not None:
            self.records = split_tracks(self.records, seed)[split]
        if not self.records:
            raise ValueError(f"Empty dataset for split {split!r}")
        if len({str(Path(r.audio_path).resolve()) for r in self.records}) != len(self.records):
            raise ValueError("Duplicate audio records")
        self.cache_dir = external_path(cache_dir)
        self.context_frames, self.config, self.split = context_frames, config, split
        self.lru_size = lru_size
        self._lru = OrderedDict()
        self._offsets = [0]
        self._identities = []
        for record in self.records:
            payload = self._read_labels(record)
            identity = {"audio": file_identity(record.audio_path),
                        "labels": file_identity(record.labels_path), "group": record.group_id}
            self._identities.append(identity)
            self._offsets.append(self._offsets[-1] + len(payload["labels"]))
        self.group_ids = frozenset(r.group_id for r in self.records)
        self.checkpoint_signature = _digest({"records": self._identities,
            "config": asdict(config), "context": context_frames, "split": split})

    def _read_labels(self, record: TrackRecord) -> dict:
        try:
            payload = torch.load(record.labels_path, weights_only=True, mmap=True, map_location="cpu")
            if (payload["version"] != VERSION or payload["audio_config"] != asdict(self.config)
                    or payload["center"] is not True or payload["string_order"] != list(STRING_ORDER)
                    or payload["silence_class"] != SILENCE or payload["group_id"] != record.group_id
                    or payload["audio"] != file_identity(record.audio_path)):
                raise ValueError("Labels metadata does not match this audio/config/group")
            if payload["annotation"] != file_identity(payload["annotation"]["path"]):
                raise ValueError("Source annotation changed; prepare labels again")
            labels = payload["labels"]
            if (labels.dtype != torch.long or labels.ndim != 2 or labels.shape[1] != 6
                    or len(labels) == 0 or int(labels.min()) < 0 or int(labels.max()) > 25):
                raise ValueError("Expected nonempty long [T, 6] labels in 0..25")
            if len(payload["tuning"]) != 6:
                raise ValueError("Expected six tuning values")
            return payload
        except Exception as exc:
            raise ValueError(f"Invalid labels {record.labels_path}: {exc}") from exc

    def assert_unchanged(self) -> None:
        for record, identity in zip(self.records, self._identities):
            if file_identity(record.audio_path) != identity["audio"] or file_identity(record.labels_path) != identity["labels"]:
                raise ValueError(f"Dataset changed after initialization: {record.audio_path}")
            self._read_labels(record)

    def __len__(self):
        return self._offsets[-1]

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        track = bisect.bisect_right(self._offsets, index) - 1
        frame = index - self._offsets[track]
        record = self.records[track]
        identity = self._identities[track]
        if file_identity(record.audio_path) != identity["audio"] or file_identity(record.labels_path) != identity["labels"]:
            raise ValueError(f"Dataset changed during iteration: {record.audio_path}")
        if track not in self._lru:
            labels = self._read_labels(record)["labels"]
            features = cached_features(record.audio_path, self.cache_dir, self.config)
            if len(features) != len(labels):
                raise ValueError(f"Audio/labels frame mismatch: {record.audio_path}: {len(features)} != {len(labels)}")
            self._lru[track] = (features, labels)
            if len(self._lru) > self.lru_size:
                self._lru.popitem(last=False)
        self._lru.move_to_end(track)
        features, labels = self._lru[track]
        left = self.context_frames // 2
        start, stop = frame - left, frame - left + self.context_frames
        result = torch.full((self.context_frames, self.config.n_mels), -5.0)
        low, high = max(start, 0), min(stop, len(features))
        result[low - start:high - start] = features[low:high]
        return result, labels[frame].clone()


def prepare_synthtab(audio_root: str | Path, jams_root: str | Path, output_dir: str | Path,
                     seed: int = 42, config: AudioConfig = AudioConfig()) -> dict:
    """Prepare available audio only, keeping group assignments stable across chunks."""
    audio_root, jams_root = Path(audio_root).resolve(), Path(jams_root).resolve()
    output_dir = external_path(output_dir)
    if not audio_root.is_dir() or not jams_root.is_dir():
        raise ValueError("audio_root and jams_root must be existing directories")
    records, skipped = [], []
    for audio in sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}):
        try:
            candidates = []
            song = None
            for parent in audio.parents:
                if parent == audio_root.parent:
                    break
                candidate = jams_root / parent.name
                if candidate.is_dir():
                    candidates = sorted(candidate.glob("*.jams"))
                    song = parent.name
                    break
            if len(candidates) != 1 or song is None:
                raise ValueError(f"Expected one matching JAMS, found {len(candidates)}")
            group = re.split(r"__\d+", song, maxsplit=1)[0]
            name = _digest(str(audio.resolve())) + ".pt"
            record = prepare_track(audio, candidates[0], group, output_dir / "labels" / name,
                                   output_dir / "features", config)
            records.append(record)
        except (OSError, ValueError) as exc:
            skipped.append({"audio_path": str(audio), "reason": str(exc)})
    manifest = {"version": VERSION, "seed": seed, "ratios": [0.8, 0.1, 0.1],
                "audio_config": asdict(config), "cache_dir": str(output_dir / "features"),
                "records": [dict(asdict(r), split=split_for_group(r.group_id, seed)) for r in records],
                "skipped": skipped}
    atomic_json_save(manifest, output_dir / "manifest.json")
    if not records:
        raise ValueError(f"No supported tracks; see {output_dir / 'manifest.json'}")
    return manifest


def datasets_from_manifest(path: str | Path, context_frames: int = 1) -> dict[str, GuitarDataset]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["version"] != VERSION or manifest["ratios"] != [0.8, 0.1, 0.1]:
        raise ValueError("Unsupported manifest")
    records = []
    for item in manifest["records"]:
        if item["split"] != split_for_group(item["group_id"], manifest["seed"]):
            raise ValueError("Manifest split differs from stable group assignment")
        records.append(TrackRecord(**{key: item[key] for key in ("audio_path", "labels_path", "group_id")}))
    groups = split_tracks(records, manifest["seed"])
    return {name: GuitarDataset(items, manifest["cache_dir"], config=AudioConfig(**manifest["audio_config"]),
                               context_frames=context_frames, split=name, seed=manifest["seed"])
            for name, items in groups.items() if items}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--jams-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare_synthtab(args.audio_root, args.jams_root, args.output_dir)
    print(json.dumps({"prepared": len(result["records"]), "skipped": len(result["skipped"]),
                      "splits": {s: sum(r["split"] == s for r in result["records"]) for s in ("train", "val", "test")}}))
