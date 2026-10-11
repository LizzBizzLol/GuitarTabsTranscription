"""Acceptance suite; toy architectures are confined to tests and fixtures are temporary."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torchaudio
import audio_utils
from audio_utils import AudioConfig, compute_log_mel, atomic_torch_save
from dataset import (GuitarDataset, TrackRecord, prepare_track, jams_to_labels,
                     split_tracks, split_for_group, prepare_synthtab, datasets_from_manifest)
from train_utils import GuitarTabLoss, train_model, evaluate_model, ResumableBatchSampler

torch.set_num_threads(1)


def jam(notes=None, tempos=None):
    notes = notes or {}
    return {"annotations": [
        {"namespace": "note_tab", "sandbox": {"string_index": i, "open_tuning": tuning},
         "data": [{"time": onset, "duration": duration, "value": {"fret": fret}}
                  for onset, duration, fret in notes.get(i, [])]}
        for i, tuning in enumerate([64, 59, 55, 50, 45, 40], 1)
    ] + [{"namespace": "tempo", "data": tempos or [{"time": 0, "duration": 10000, "value": 120}]}]}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(256, 156)

    def forward(self, x):
        output = self.linear(self.dropout(x[:, 0]))
        if self.training:
            output = output * (0.95 + 0.01 * random.random() + 0.01 * np.random.random())
        return output.reshape(-1, 6, 26)


class SilenceModel(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x), 6, 26)
        logits[..., 25] = 100
        return logits


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="guitar-iteration0-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def annotation(self, value, name="notes.jams"):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def record(self, group, name="audio", samples=2048):
        wav = self.root / (name + ".wav")
        time = torch.arange(samples) / 16000
        torchaudio.save(str(wav), (0.1 * torch.sin(2 * torch.pi * 110 * time)).unsqueeze(0), 16000)
        annotation = self.annotation(jam({6: [(0, 80, 24)], 1: [(40, 80, 0)]}), name + ".jams")
        return prepare_track(wav, annotation, group, self.root / (name + ".pt"), self.root / "cache")

    def data(self):
        train = GuitarDataset([self.record("train-composition", "train")], self.root / "cache")
        val = GuitarDataset([self.record("val-composition", "val")], self.root / "cache")
        return train, val

    def model(self):
        torch.manual_seed(99)
        model = ToyModel()
        return model, torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_audio_formula_resampling_and_failures(self):
        stereo = torch.stack([torch.linspace(-0.2, 0.2, 1000), torch.linspace(0.2, -0.1, 1000)])
        result = compute_log_mel(stereo, 32000)
        mono = torchaudio.functional.resample(stereo.mean(0), 32000, 16000)
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048,
            win_length=2048, hop_length=256, n_mels=256, power=1., center=True,
            pad_mode="constant", f_min=0., f_max=8000.)
        expected = torch.log10(transform(mono) + 1e-5).T.contiguous()
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape[1], 256)
        self.assertTrue(torch.equal(compute_log_mel(torch.zeros(1), 16000), torch.full((1, 256), -5.)))
        for bad in [torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])]:
            with self.assertRaises(ValueError):
                compute_log_mel(bad, 16000, source="invalid.wav")

    def test_jams_tempo_boundaries_order_and_conflicts(self):
        tempos = [{"time": 0, "duration": 960, "value": 120},
                  {"time": 960, "duration": 960, "value": 60}]
        # first string note .016 <= t < .032; bass note .5 <= t < 1.5
        path = self.annotation(jam({1: [(30.72, 30.72, 0)], 6: [(960, 960, 24)]}, tempos))
        labels, tuning = jams_to_labels(path, 100)
        self.assertEqual(tuning, [40, 45, 50, 55, 59, 64])
        self.assertEqual(labels[0, 5], 25)
        self.assertEqual(labels[1, 5], 0)
        self.assertEqual(labels[2, 5], 25)
        self.assertEqual(labels[31, 0], 25)
        self.assertEqual(labels[32, 0], 24)
        self.assertEqual(labels[93, 0], 24)
        self.assertEqual(labels[94, 0], 25)
        path = self.annotation(jam({6: [(0, 200, 3), (100, 300, 3)]}))
        self.assertEqual(jams_to_labels(path, 30)[0][12, 0], 3)
        for value in [jam({6: [(0, 200, 3), (100, 300, 4)]}), jam({6: [(0, 200, 25)]}),
                      jam(tempos=[{"time": 10, "duration": 20, "value": 120}])]:
            path = self.annotation(value)
            with self.assertRaisesRegex(ValueError, "notes.jams"):
                jams_to_labels(path, 30)

    def test_splits_stable_grouped(self):
        records = [TrackRecord(str(i) + "a", "labels", "song" + str(i)) for i in range(1000)]
        variants = [TrackRecord(str(i) + "b", "labels", "song" + str(i)) for i in range(1000)]
        initial = split_tracks(records)
        later = split_tracks(records + variants + [TrackRecord("new", "labels", "new-song")])
        groups = {s: {r.group_id for r in initial[s]} for s in initial}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        for split, items in initial.items():
            self.assertTrue(all(r in later[split] for r in items))
            self.assertTrue(all(split_for_group(r.group_id) == split for r in later[split]))
        self.assertTrue(750 < len(initial["train"]) < 850)

    def test_dataset_context_mmap_mismatch_and_cache(self):
        record = self.record("composition")
        default = GuitarDataset([record], self.root / "cache")
        even = GuitarDataset([record], self.root / "cache", context_frames=4, lru_size=1)
        features, targets = default[0]
        self.assertEqual(features.shape, (1, 256))
        self.assertEqual(targets.shape, (6,))
        self.assertEqual(targets.dtype, torch.long)
        ctx, target = even[0]
        self.assertTrue(torch.equal(ctx[:2], torch.full((2, 256), -5.)))
        self.assertTrue(torch.equal(ctx[2], features[0]))
        self.assertTrue(torch.equal(target, targets))
        last, _ = even[len(even) - 1]
        self.assertTrue(torch.equal(last[-1], torch.full((256,), -5.)))
        cached = list((self.root / "cache").glob("*.pt"))
        identity = [(p.name, p.stat().st_mtime_ns) for p in cached]
        other = GuitarDataset([record], self.root / "cache")
        other[0]
        self.assertEqual(identity, [(p.name, p.stat().st_mtime_ns) for p in cached])
        payload = torch.load(record.labels_path, weights_only=True)
        payload["labels"] = payload["labels"][:-1]
        atomic_torch_save(payload, record.labels_path)
        with self.assertRaises(ValueError):
            default[0]
        mismatch = GuitarDataset([record], self.root / "cache")
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            mismatch[0]
        with self.assertRaises(ValueError):
            GuitarDataset([record], self.root / "cache", config=AudioConfig(hop_length=512))

    def test_loss_exact_backward_and_silence(self):
        logits = torch.randn(5, 6, 26, requires_grad=True)
        targets = torch.randint(0, 26, (5, 6))
        targets[0] = 25
        weights = torch.ones(26)
        weights[25] = 0.1
        reference = sum(torch.nn.functional.cross_entropy(logits[:, s], targets[:, s],
                          weight=weights, reduction="none").mean() for s in range(6))
        loss = GuitarTabLoss()(logits, targets)
        self.assertTrue(torch.equal(loss, reference))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        silent = torch.full((5, 6), 25)
        self.assertTrue(torch.allclose(GuitarTabLoss()(logits, silent), GuitarTabLoss(1.0)(logits, silent) * 0.1))
        for shape in [(5, 26, 6), (5, 6, 25)]:
            with self.assertRaises(ValueError):
                GuitarTabLoss()(torch.zeros(shape), targets)

    def test_validation_all_silence(self):
        train, val = self.data()
        result = evaluate_model(SilenceModel(), val, GuitarTabLoss(), batch_size=3)
        self.assertEqual(result["active_f1"], 0)
        self.assertEqual(result["non_silent_accuracy"], 0)
        self.assertEqual(result["pred_silence_ratio"], 1)
        self.assertGreater(result["false_negative"], 0)
        self.assertLess(result["ref_silence_ratio"], 1)

    def test_exact_mid_epoch_resume(self):
        train, val = self.data()
        full, full_optimizer = self.model()
        full_trace = []
        full_history = train_model(full, full_optimizer, train, val, num_epochs=3, batch_size=4,
            checkpoint_dir=self.root / "full", device="cpu", on_batch=lambda *x: full_trace.append(x))
        partial, partial_optimizer = self.model()
        partial_trace = []
        def interrupt(*values):
            partial_trace.append(values)
            if values[2] == 2:
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            train_model(partial, partial_optimizer, train, val, num_epochs=3, batch_size=4,
                checkpoint_dir=self.root / "partial", device="cpu", on_batch=interrupt)
        saved = torch.load(self.root / "partial/last_checkpoint.pt", weights_only=True)
        self.assertEqual((saved["epoch"], saved["next_batch"], saved["global_step"]), (0, 2, 2))
        resumed, resumed_optimizer = self.model()
        resumed_history = train_model(resumed, resumed_optimizer, train, val, num_epochs=3, batch_size=4,
            checkpoint_dir=self.root / "partial", device="cpu", on_batch=lambda *x: partial_trace.append(x))
        self.assertEqual(full_trace, partial_trace)
        self.assertEqual(full_history, resumed_history)
        for name, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
        for parameter, state in full_optimizer.state_dict()["state"].items():
            for name, value in state.items():
                other = resumed_optimizer.state_dict()["state"][parameter][name]
                self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
        self.assertEqual(list(ResumableBatchSampler(17, 4, 42, 0))[2:],
                         list(ResumableBatchSampler(17, 4, 42, 0, 2)))

    def test_checkpoint_failure_corruption_and_configuration(self):
        train, val = self.data()
        model, optimizer = self.model()
        folder = self.root / "checkpoint"
        train_model(model, optimizer, train, val, num_epochs=1, batch_size=4, checkpoint_dir=folder, device="cpu")
        path = folder / "last_checkpoint.pt"
        original = path.read_bytes()
        with patch("audio_utils.os.replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                train_model(model, optimizer, train, val, num_epochs=2, batch_size=4, checkpoint_dir=folder, device="cpu")
        self.assertEqual(original, path.read_bytes())
        self.assertFalse(list(folder.glob("*.tmp")))
        for changed in [{"batch_size": 2}, {"silence_weight": 0.3}]:
            args = dict(num_epochs=2, batch_size=4, checkpoint_dir=folder, device="cpu")
            args.update(changed)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                train_model(model, optimizer, train, val, **args)
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Cannot restore"):
            train_model(model, optimizer, train, val, num_epochs=2, batch_size=4, checkpoint_dir=folder, device="cpu")
        val.split = "test"
        with self.assertRaisesRegex(ValueError, "reserved"):
            train_model(model, optimizer, train, val, num_epochs=1, checkpoint_dir=self.root / "test")

    def test_epoch_boundary_resume_and_completed_budget(self):
        train, val = self.data()
        full, opt = self.model()
        expected = train_model(full, opt, train, val, num_epochs=2, batch_size=4,
                               checkpoint_dir=self.root / "full", device="cpu")
        stopped, stopped_opt = self.model()
        with patch("train_utils.evaluate_model", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                train_model(stopped, stopped_opt, train, val, num_epochs=2, batch_size=4,
                            checkpoint_dir=self.root / "partial", device="cpu")
        restored, restored_opt = self.model()
        actual = train_model(restored, restored_opt, train, val, num_epochs=2, batch_size=4,
                             checkpoint_dir=self.root / "partial", device="cpu")
        self.assertEqual(expected, actual)
        for key in full.state_dict():
            self.assertTrue(torch.equal(full.state_dict()[key], restored.state_dict()[key]))
        checkpoint = self.root / "partial/last_checkpoint.pt"
        previous = checkpoint.read_bytes()
        train_model(restored, restored_opt, train, val, num_epochs=2, batch_size=4,
                    checkpoint_dir=self.root / "partial", device="cpu",
                    on_batch=lambda *args: self.fail("Completed budget must not train"))
        self.assertEqual(previous, checkpoint.read_bytes())

    def test_leakage_and_changed_sources_are_rejected(self):
        train, val = self.data()
        model, opt = self.model()
        val.group_ids = train.group_ids
        with self.assertRaisesRegex(ValueError, "leakage"):
            train_model(model, opt, train, val, num_epochs=1, checkpoint_dir=self.root / "leak")
        val.group_ids = frozenset(["val-composition"])
        record = train.records[0]
        payload = torch.load(record.labels_path, weights_only=True)
        annotation = Path(payload["annotation"]["path"])
        annotation.write_text(annotation.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "annotation changed"):
            train_model(model, opt, train, val, num_epochs=1, checkpoint_dir=self.root / "changed")

    def test_manifest_same_score_variants_and_rejections(self):
        audio_root = self.root / "audio"
        jams_root = self.root / "jams"
        for part in [1, 2]:
            song = f"example - gp3__{part} - guitar__midi"
            annotation_dir = jams_root / song
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "notes.jams").write_text(json.dumps(jam()), encoding="utf-8")
            for timbre in ["clean", "distortion"]:
                directory = audio_root / "electric_clean" / timbre / song
                directory.mkdir(parents=True)
                torchaudio.save(str(directory / "mic.wav"), torch.zeros(1, 512), 16000)
        unsupported = audio_root / "unsupported.wav"
        torchaudio.save(str(unsupported), torch.zeros(1, 512), 16000)
        manifest = prepare_synthtab(audio_root, jams_root, self.root / "prepared")
        self.assertEqual(len(manifest["records"]), 4)
        self.assertEqual(len(manifest["skipped"]), 1)
        self.assertEqual(len({r["group_id"] for r in manifest["records"]}), 1)
        self.assertEqual(len({r["split"] for r in manifest["records"]}), 1)
        datasets = datasets_from_manifest(self.root / "prepared/manifest.json")
        self.assertEqual(sum(len(d.records) for d in datasets.values()), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
