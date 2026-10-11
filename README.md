# GuitarTabsTranscription

Iterations 0 through 1.2: plain PyTorch infrastructure and single-frame frequency models
for guitar audio-to-tablature research. No full training experiment has been run.
The reference implementation is https://github.com/LizzBizzLol/PP-VKL-GuitarTabs.

## Environment

Python 3.11 is the verified runtime. Install matching torch/torchaudio versions:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA CUDA 12.1 wheels, install torch and torchaudio from the official index:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

`torchaudio.load` uses an installed audio backend (soundfile is included).
Model construction and optimizer creation belong to the caller. Move the model
to its device before constructing the optimizer. Seed model initialization in
the caller when reproducing a fresh experiment; the training seed controls
sampling and training RNG after that initialization.

## Data contract

- Features: amplitude Mel-STFT, mono, 16000 Hz, n_fft=win_length=2048,
  hop_length=256, n_mels=256, Hann window, center=True, constant padding,
  HTK Mel scale, 0..8000 Hz, power=1.0; log10(amplitude + 1e-5).
- A frame is spaced 16 ms apart; the FFT-bin spacing is 7.8125 Hz.
- A sample has float32 features `[context_frames, 256]` and long targets `[6]`.
- Target classes: frets 0..24; silence 25. String order: 6,5,4,3,2,1
  (low E to high E in standard tuning). Alternate tuning stays in metadata.
- Context defaults to one frame. For even widths the target is the frame at
  index `context_frames // 2`. Out-of-track context is filled with -5.
- `.pt` label files contain `labels [T,6]`, the audio configuration, center,
  string_order, silence_class, tuning, group_id, audio and annotation identities.
  Old CQT/21-class label caches are incompatible and must be regenerated.

SynthTab `note_tab` JAMS time/duration values are MIDI ticks, not seconds.
The converter integrates the tempo map at 960 ticks per quarter note and samples
half-open `[onset, offset)` intervals at frame-center times. Same-fret overlaps
are merged. Different-fret overlaps, invalid tempo coverage, unsupported frets
and non-six-string annotations are rejected with file-specific diagnostics.
Very short notes containing no frame center are not represented at this grid.

## Preparing the existing SynthTab chunk

All generated data must be stored outside this repository. For example:

```powershell
python dataset.py --audio-root D:/DATA/SynthTab_Full/current_chunk --jams-root D:/DATA/SynthTab_Full/jams_midi/outall --output-dir D:/DATA/GuitarTabsTranscription/iteration0
```

This prepares each available recording and writes labels, features and a
`manifest.json` with explicit prepared/skipped records and rejection reasons.
Inspect the skipped list before using a subset. Corrupt source audio is rejected,
not repaired. No preparation is performed during import.

For SynthTab, the source-score name before `__<part number>` is the group_id.
All instrument parts, timbres and microphones of that score share a split.
Separate copies/versions of the same composition under different source-score
names need manually unified group_ids; names alone cannot detect all duplicates.
Custom data uses explicitly supplied TrackRecord group_ids.

The stable SHA-256 assignment uses seed 42 and approximate 80/10/10 thresholds.
It never splits frames. Adding a chunk preserves prior group assignments.
Small chunks may have empty splits: the loader omits those splits; training
requires nonempty train and val. Include more compositions instead of moving
groups between splits. Historical baseline scores used a different split and
feature grid and are not directly comparable to future scores on this setup.

```python
from dataset import datasets_from_manifest

datasets = datasets_from_manifest(
    "D:/DATA/GuitarTabsTranscription/iteration0/manifest.json",
    context_frames=1,
)
train_dataset, val_dataset = datasets["train"], datasets["val"]
```

Alternatively, use `prepare_track(audio_path, jams_path, group_id, labels_path,
cache_dir)` and construct `GuitarDataset(records, cache_dir, split="train")`.
Each feature tensor is cached atomically once. Memory-mapped tensors and a bounded
LRU avoid loading the full dataset into RAM. Preparation needs memory for one
whole track. Frame-count/config/audio mismatches fail instead of truncating.
File identity uses absolute path, size and nanosecond modification time; do not
modify files while preserving their timestamps. Regenerate after relocating data.

## Frequency U-Net models (iteration 1)

```python
from models import FullFrequencyUNet, TruncatedFrequencyUNet
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FullFrequencyUNet().to(device)
# Alternative: TruncatedFrequencyUNet(pool_bins=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Both models take `[B,1,256]` from `GuitarDataset(context_frames=1)` and return
raw logits `[B,6,26]`, strings 6 to 1. All convolutions operate on frequency;
multiple time-context frames are rejected, not treated as input channels.
The common encoder has three Conv1d(3, padding=1)-BatchNorm-ReLU blocks with
channels 32/64/128 and MaxPool1d(2) after each. Pre-pool skip widths are
256/128/64, and the final pooled bottleneck width is 32.

The full model upsamples with three ConvTranspose1d(2, stride=2) layers,
concatenates corresponding skips, and applies Conv-BN-ReLU blocks with
128/64/32 channels. Flatten produces 8192 features for six separate Linear heads.
The truncated model has no decoder: AdaptiveAvgPool1d(8) on each skip produces
1792 concatenated features for its six heads. Pooling to eight positions retains
coarse frequency location, unlike global averaging. Constructors accept
`n_mels` (a multiple of eight) and three positive `channels`; `pool_bins` must be
positive and not exceed the deepest skip width (`n_mels // 4`). Encoder weights
are independent between model instances. Changing architecture requires a new
checkpoint directory. Use the `train_model` call below without adapters.

## Additive frequency encoder (iteration 1.1)

```python
from models import AdditiveFrequencyEncoder

model = AdditiveFrequencyEncoder(n_mels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

The input/output contract and training API are unchanged. Conv-BN-ReLU blocks
use fixed channels 64/128/256, kernel 3 and padding 1. Only the first two blocks
are followed by MaxPool1d(2). Pooled skips have shapes `[B,64,128]` and
`[B,128,64]`; the third block produces `[B,256,64]`. The 1x1 projections
`proj1_to_3` (64 to 256, stride 2) and `proj2_to_3` (128 to 256, stride 1)
align these skips with the third output. Their elementwise sum is flattened
without extra pooling and sent to six independent 26-class Linear heads.

Addition has no trainable parameters and retains 256 channels. Concatenating
the same three projected maps would produce 768 channels: the six heads would
need 7,667,712 weights instead of 2,555,904, with 156 biases in either case.
The projections themselves add 49,664 parameters including biases. The complete
default model has **2,730,140 parameters**, more than either earlier U-Net;
addition is cheaper than matched-map concatenation, not necessarily cheaper than
a different architecture. Use a separate checkpoint directory for this model.
`n_mels` must be a positive multiple of four. As with ordinary BatchNorm,
training needs more than one value per channel: the smallest `n_mels=4` requires
batch size greater than one, though batch-one inference works in eval mode.

## Sequential baseline (iteration 1.2)

```python
from models import TabCNNBaseline

model = TabCNNBaseline(n_mels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

This is the specified **1D Log-Mel baseline**, not an exact reproduction of the
original TabCNN. It has no skip connections, projections or branch merging.
The first two blocks are Conv1d(3, padding=1)-BatchNorm-ReLU-Dropout(0.25),
with channels 32 then 64 and unchanged frequency width. The third block is
Conv1d(64,128,3,padding=1)-MaxPool1d(2)-BatchNorm-ReLU-Dropout(0.25), in that
order. Flatten of `[B,128,128]` gives 16384 features for six independent heads.
`n_mels` must be positive and even; at the minimum width of two, training needs
batch size greater than one for BatchNorm. Dropout is disabled in eval mode.

All four models have the same `forward(x)` contract and work directly with
`GuitarDataset(context_frames=1)`, `GuitarTabLoss` and `train_model`.
Defaults, including all trainable weights and biases:

| Model | Input | Raw logits | Parameters |
| --- | --- | --- | ---: |
| FullFrequencyUNet | `[B,1,256]` | `[B,6,26]` | 1,492,764 |
| TruncatedFrequencyUNet | `[B,1,256]` | `[B,6,26]` | 311,196 |
| AdditiveFrequencyEncoder | `[B,1,256]` | `[B,6,26]` | 2,730,140 |
| TabCNNBaseline | `[B,1,256]` | `[B,6,26]` | 2,587,548 |

Exact CPU mid-epoch resume is tested for all four default architectures with
Adam, including Dropout RNG and BatchNorm buffers. Each architecture needs its
own checkpoint directory. These smoke tests are not real-data quality benchmarks.

## Loss and training

Supply an ordinary torch.nn.Module returning **raw logits `[B,6,26]`**;
do not apply softmax. The loss is the sum of six independent CrossEntropyLoss
terms, each averaged over the batch after per-class weighting. We use a fixed
batch denominator, so silence weighting also reduces all-silent batch gradients.
Default silence_weight=0.1 and fret weights=1 are initial settings, not a claim
that silence collapse has been eliminated for an untested architecture.
Keep all silent frames and select weights using validation only.

```python
from train_utils import train_model

# model and optimizer are constructed by the caller.
history = train_model(
    model, optimizer, train_dataset, val_dataset,
    num_epochs=10, batch_size=64, device="auto",
    checkpoint_dir="D:/DATA/GuitarTabsTranscription/runs/my_experiment",
)
```

Validation reports weighted loss, micro F1 over active string/fret positions,
accuracy restricted to active reference targets and reference/predicted silence
ratios. Wrong frets count as both a false positive and false negative. All-silent
prediction on active targets gives active F1=0, even with high overall accuracy.
Empty active sets produce F1=0. Test is reserved for final evaluation and is
rejected as an input to train_model. Data groups must be disjoint.

## Checkpoints

Every successful optimizer step and completed epoch atomically replaces
`last_checkpoint.pt`. The file contains model/optimizer states, zero-based epoch,
next_batch, global_step, current_loss, partial-epoch statistics, validation
history, Python/NumPy/Torch/CUDA RNG states, loss configuration and run signature.
The signature binds datasets, context, batch size, shuffle seed, model structure,
optimizer settings and device. num_epochs is a total budget and may be increased
on resume. A completed budget returns the saved history without additional steps.

Resume happens automatically if the file exists. A deterministic batch sampler
reconstructs the epoch permutation and begins at next_batch; it does not replay
earlier optimizer steps. num_workers=0 and independent loader generators isolate
loading from model RNG. `on_batch(epoch,batch,global_step,loss)` can log or interrupt
after a committed checkpoint; it must not alter the model, optimizer or RNG.
No scheduler, AMP, random augmentation, gradient accumulation or distributed
training is provided in iteration 0.

Interruption during a batch recovers the last persisted optimizer step. Saving
errors stop training and preserve the prior checkpoint. Corrupt/incompatible
checkpoints raise an error; they never silently restart training. Use a different
run directory for a different model/data setup. Exact CPU resume is verified;
GPU operations may be nondeterministic, and bitwise equivalence across devices
or library versions is not promised. Use trusted `.pt` files only (weights-only
loading is enabled).

## Inspecting a frame and feature maps

```python
from visualizer import plot_spectrogram_frame, extract_feature_maps, plot_feature_maps

frame, targets = train_dataset[0]  # [1,256], [6]
fig, ax = plot_spectrogram_frame(frame)
fig.savefig("D:/DATA/GuitarTabsTranscription/input_frame.png", dpi=150)
maps = extract_feature_maps(model, frame.unsqueeze(0).to(device))
fig, axes = plot_feature_maps(maps, sample_index=0)
fig.savefig("D:/DATA/GuitarTabsTranscription/feature_maps.png", dpi=150)
```

Functions do not call `show` or save automatically. Close returned figures with
`matplotlib.pyplot.close(fig)` when done; for headless execution select Matplotlib
`Agg` before importing the visualizer. Default hooks capture `encoder.0` and
`encoder.2` after ReLU: `[B,32,256]` and `[B,128,64]` for the U-Nets;
`[B,64,256]` and `[B,256,64]` for AdditiveFrequencyEncoder. The additive model
has no pooling after its third block.
For TabCNNBaseline the corresponding maps are `[B,32,256]` and `[B,128,128]`;
hooks see the complete blocks, including Dropout (disabled during extraction).
Pass distinct names from `model.named_modules()` via `layer_names` to inspect
other tensor-valued `[B,C,F]` layers. The input must already match the model device.
Extraction uses eval/no-grad, removes hooks even on failure, restores each
module's prior mode and returns independent CPU copies. Each heatmap has its own
activation scale; the horizontal axis is feature-map position, not physical Hz.
Random weights do not establish tonic or overtone detectors. Such interpretation
requires trained weights and comparison against annotated audio.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The acceptance suite uses disposable audio/JAMS fixtures and small test-only
models. It checks feature math, tempo changes and note boundaries, whole-score
splits, context padding, metadata mismatches, active metrics, exact CPU resume
with dropout and Python/NumPy randomness, epoch-boundary interruption, corrupted
checkpoints and atomic-save failure. Test architectures are not production models.
