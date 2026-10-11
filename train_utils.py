"""Six-string loss, validation and exact committed-batch training resume."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from audio_utils import atomic_torch_save, external_path


class GuitarTabLoss(nn.Module):
    def __init__(self, silence_weight: float = 0.1):
        super().__init__()
        if not math.isfinite(silence_weight) or silence_weight <= 0:
            raise ValueError("silence_weight must be finite and positive")
        self.silence_weight = float(silence_weight)
        weights = torch.ones(26)
        weights[25] = silence_weight
        self.criteria = nn.ModuleList([nn.CrossEntropyLoss(weight=weights.clone(), reduction="none") for _ in range(6)])

    def forward(self, logits, targets):
        if logits.ndim != 3 or logits.shape[1:] != (6, 26) or targets.shape != logits.shape[:2]:
            raise ValueError("Expected logits [B, 6, 26] and targets [B, 6]")
        if len(targets) == 0 or targets.dtype != torch.long or int(targets.min()) < 0 or int(targets.max()) > 25:
            raise ValueError("Expected nonempty long targets in 0..25")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model logits")
        return sum(criterion(logits[:, string], targets[:, string]).mean()
                   for string, criterion in enumerate(self.criteria))


class ResumableBatchSampler(Sampler):
    def __init__(self, length: int, batch_size: int, seed: int, epoch: int, next_batch: int = 0):
        self.length, self.batch_size = length, batch_size
        self.seed, self.epoch, self.next_batch = seed, epoch, next_batch
        if length <= 0 or batch_size <= 0 or not 0 <= next_batch <= math.ceil(length / batch_size):
            raise ValueError("Invalid batch sampler size/cursor")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.length, generator=generator)
        for start in range(self.next_batch * self.batch_size, self.length, self.batch_size):
            yield order[start:start + self.batch_size].tolist()

    def __len__(self):
        return math.ceil(self.length / self.batch_size) - self.next_batch


def _rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed; exact continuation is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def restore_checkpoint(path, model, optimizer, *, signature: str, device) -> dict | None:
    """Restore a trusted tensor/primitive checkpoint, rejecting incompatible runs."""
    path = external_path(path)
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        required = {"version", "signature", "model_state_dict", "optimizer_state_dict", "epoch",
                    "next_batch", "global_step", "current_loss", "rng_state", "loss_config",
                    "history", "epoch_loss_sum", "epoch_samples"}
        if not required <= state.keys() or state["version"] != 1 or state["signature"] != signature:
            raise ValueError("Checkpoint schema or dataset/model/training configuration changed")
        for key in ("epoch", "next_batch", "global_step", "epoch_samples"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Invalid checkpoint {key}")
        if not math.isfinite(state["current_loss"]) or not math.isfinite(state["epoch_loss_sum"]):
            raise ValueError("Non-finite checkpoint loss")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for parameter, values in optimizer.state.items():
            optimizer.state[parameter] = _to_device(values, device)
        _restore_rng(state["rng_state"])
        return state
    except Exception as exc:
        raise ValueError(f"Cannot restore checkpoint {path}: {exc}") from exc


@torch.no_grad()
def evaluate_model(model, dataset, criterion, *, batch_size=64, device="cpu", seed=42) -> dict:
    if len(dataset) == 0:
        raise ValueError("Empty validation dataset")
    was_training = model.training
    model.eval()
    total_loss = examples = correct_active = ref_active = pred_active = true_positive = 0
    ref_silent = pred_silent = entries = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    try:
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            predictions = logits.argmax(dim=-1)
            reference_active, predicted_active = targets != 25, predictions != 25
            exact = predictions == targets
            hits = int((exact & reference_active).sum())
            true_positive += hits
            correct_active += hits
            ref_active += int(reference_active.sum())
            pred_active += int(predicted_active.sum())
            ref_silent += int((~reference_active).sum())
            pred_silent += int((~predicted_active).sum())
            entries += targets.numel()
            examples += len(targets)
            total_loss += float(loss) * len(targets)
    finally:
        model.train(was_training)
    return {"loss": total_loss / examples,
            "active_f1": 2 * true_positive / (ref_active + pred_active) if ref_active + pred_active else 0.0,
            "non_silent_accuracy": correct_active / ref_active if ref_active else 0.0,
            "pred_silence_ratio": pred_silent / entries, "ref_silence_ratio": ref_silent / entries,
            "true_positive": true_positive, "false_positive": pred_active - true_positive,
            "false_negative": ref_active - true_positive, "examples": examples}


def train_model(model, optimizer, train_dataset, val_dataset, *, num_epochs: int,
                checkpoint_dir, batch_size: int = 64, device="auto", seed: int = 42,
                silence_weight: float = 0.1, on_batch=None) -> list[dict]:
    """Train to a total epoch budget. on_batch is a non-random logging hook."""
    if type(num_epochs) is not int or num_epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("num_epochs and batch_size must be positive integers")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation must be nonempty")
    if train_dataset.split not in (None, "train") or val_dataset.split not in (None, "val"):
        raise ValueError("Training requires train/val datasets; test is reserved for final evaluation")
    if train_dataset.group_ids & val_dataset.group_ids:
        raise ValueError("Source-composition leakage between train and validation")
    train_dataset.assert_unchanged()
    val_dataset.assert_unchanged()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model.to(device)
    criterion = GuitarTabLoss(silence_weight).to(device)
    optimizer_config = [{key: value for key, value in group.items() if key != "params"}
                        for group in optimizer.param_groups]
    description = {"train": train_dataset.checkpoint_signature, "val": val_dataset.checkpoint_signature,
                   "model": repr(model), "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                   "optimizer_config": optimizer_config, "batch_size": batch_size, "seed": seed,
                   "silence_weight": silence_weight, "device": str(device), "num_workers": 0}
    signature = hashlib.sha256(json.dumps(description, sort_keys=True, default=str).encode()).hexdigest()
    path = external_path(checkpoint_dir) / "last_checkpoint.pt"
    state = restore_checkpoint(path, model, optimizer, signature=signature, device=device)
    if state is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        state = {"epoch": 0, "next_batch": 0, "global_step": 0, "current_loss": 0.0,
                 "epoch_loss_sum": 0.0, "epoch_samples": 0, "history": []}

    def save():
        atomic_torch_save({**state, "version": 1, "signature": signature,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _rng_state(), "loss_config": {"silence_weight": silence_weight},
            "training_config": description}, path)

    total_batches = math.ceil(len(train_dataset) / batch_size)
    if state["epoch"] > num_epochs:
        raise ValueError("Requested total epoch budget is smaller than completed training")
    if state["next_batch"] > total_batches:
        raise ValueError("Checkpoint cursor exceeds epoch length")
    for epoch in range(state["epoch"], num_epochs):
        train_dataset.assert_unchanged()
        val_dataset.assert_unchanged()
        model.train()
        first_batch = state["next_batch"]
        sampler = ResumableBatchSampler(len(train_dataset), batch_size, seed, epoch, first_batch)
        loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0,
                            generator=torch.Generator().manual_seed(seed + epoch))
        for batch, (features, targets) in enumerate(loader, start=first_batch):
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; last committed checkpoint preserved")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Non-finite gradients; last committed checkpoint preserved")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Non-finite parameters after optimizer step; checkpoint preserved")
            state["epoch"] = epoch
            state["next_batch"] = batch + 1
            state["global_step"] += 1
            state["current_loss"] = float(loss.detach())
            state["epoch_loss_sum"] += state["current_loss"] * len(targets)
            state["epoch_samples"] += len(targets)
            save()
            if on_batch is not None:
                on_batch(epoch, batch, state["global_step"], state["current_loss"])
        validation = evaluate_model(model, val_dataset, criterion, batch_size=batch_size, device=device, seed=seed + epoch)
        state["history"].append({"epoch": epoch, "train_loss": state["epoch_loss_sum"] / state["epoch_samples"],
                                  "validation": validation})
        state.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0.0, epoch_samples=0)
        save()
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={state['history'][-1]['train_loss']:.6f}, "
              f"val_loss={validation['loss']:.6f}, active_f1={validation['active_f1']:.4f}")
    return state["history"]
