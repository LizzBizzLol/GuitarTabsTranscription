# GuitarTabsTranscription

Iteration 0: plain PyTorch infrastructure for guitar audio-to-tablature research.
The repository contains three production Python modules and no model architecture.
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

## Verification

```powershell
python -m unittest discover -s tests -v
```

The acceptance suite uses disposable audio/JAMS fixtures and small test-only
models. It checks feature math, tempo changes and note boundaries, whole-score
splits, context padding, metadata mismatches, active metrics, exact CPU resume
with dropout and Python/NumPy randomness, epoch-boundary interruption, corrupted
checkpoints and atomic-save failure. Test architectures are not production models.
