"""Inspect Log-Mel frames and activations without changing model state."""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_spectrogram_frame(frame, *, ax=None):
    frame = torch.as_tensor(frame).detach().cpu()
    if frame.shape == (1, 256):
        frame = frame[0]
    if frame.shape != (256,) or not torch.isfinite(frame).all():
        raise ValueError("Expected a finite Log-Mel frame [256] or [1, 256]")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    ax.plot(range(256), frame.numpy(), linewidth=1)
    ax.set(xlabel="Mel bin", ylabel="log10(amplitude + 1e-5)",
           title="Input Log-Mel frame", xlim=(0, 255))
    return ax.figure, ax


def extract_feature_maps(model, inputs, *, layer_names=None):
    names = ("encoder.0", "encoder.2") if layer_names is None else tuple(layer_names)
    modules = dict(model.named_modules())
    if not names or len(set(names)) != len(names) or any(name not in modules for name in names):
        raise ValueError("layer_names must contain distinct existing module names")
    modes = [(module, module.training) for module in model.modules()]
    maps, handles = {}, []

    def hook(name):
        def capture(module, args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3 or not torch.isfinite(output).all():
                raise ValueError(f"Layer {name} must produce finite feature maps [B, C, F]")
            maps[name] = output.detach().to("cpu").clone()
        return capture

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(hook(name)))
        model.eval()
        with torch.no_grad():
            model(inputs)
        if any(name not in maps for name in names):
            raise ValueError("Some requested layers were not executed")
        return {name: maps[name] for name in names}
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training


def plot_feature_maps(feature_maps, *, sample_index=0):
    if not feature_maps or type(sample_index) is not int or sample_index < 0:
        raise ValueError("Expected nonempty feature maps and a nonnegative sample_index")
    arrays = []
    for name, maps in feature_maps.items():
        maps = torch.as_tensor(maps).detach().cpu()
        if maps.ndim != 3 or min(maps.shape) == 0 or sample_index >= maps.shape[0] or not torch.isfinite(maps).all():
            raise ValueError(f"Invalid feature maps or sample_index for {name}")
        arrays.append((name, maps[sample_index].numpy()))
    fig, axes = plt.subplots(len(arrays), 1, figsize=(10, 3 * len(arrays)),
                             squeeze=False, layout="constrained")
    for ax, (name, array) in zip(axes[:, 0], arrays):
        image = ax.imshow(array, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set(title=f"{name}: {array.shape[0]} channels x {array.shape[1]} positions",
               xlabel="Frequency position in feature map", ylabel="Channel")
        fig.colorbar(image, ax=ax, label="Activation")
    return fig, axes[:, 0]
