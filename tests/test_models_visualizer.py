"""Iteration 1 architecture, integration and non-mutating inspection checks."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from models import FullFrequencyUNet, TruncatedFrequencyUNet, AdditiveFrequencyEncoder, TabCNNBaseline
from visualizer import plot_spectrogram_frame, extract_feature_maps, plot_feature_maps
from train_utils import GuitarTabLoss, train_model
import test_infrastructure

torch.set_num_threads(1)


class ModelsAndVisualizer(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_forward_backward_and_head_sizes(self):
        for cls, feature_width in [(FullFrequencyUNet, 8192), (TruncatedFrequencyUNet, 1792)]:
            for batch in (1, 3):
                with self.subTest(model=cls.__name__, batch=batch):
                    model = cls()
                    output = model(torch.randn(batch, 1, 256))
                    self.assertEqual(output.shape, (batch, 6, 26))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(model.heads[0].in_features, feature_width)
                    GuitarTabLoss()(output, torch.randint(0, 26, (batch, 6))).backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    for head in model.heads:
                        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_encoder_decoder_shapes(self):
        full, truncated = FullFrequencyUNet(), TruncatedFrequencyUNet()
        self.assertEqual(str(full.encoder), str(truncated.encoder))
        self.assertFalse(any(isinstance(m, nn.ConvTranspose1d) for m in truncated.modules()))
        self.assertFalse(hasattr(truncated, "decoder"))
        self.assertNotEqual(full.encoder[0][0].weight.data_ptr(), truncated.encoder[0][0].weight.data_ptr())
        names = ["encoder.0", "pools.0", "encoder.1", "pools.1", "encoder.2", "pools.2",
                 "upsamples.0", "decoder.0", "upsamples.1", "decoder.1", "upsamples.2", "decoder.2"]
        maps = extract_feature_maps(full, torch.randn(2, 1, 256), layer_names=names)
        expected = [(32, 256), (32, 128), (64, 128), (64, 64), (128, 64), (128, 32),
                    (128, 64), (128, 64), (64, 128), (64, 128), (32, 256), (32, 256)]
        self.assertEqual([tuple(m.shape[1:]) for m in maps.values()], expected)
        handles, widths = [], []
        try:
            for block in full.decoder:
                handles.append(block.register_forward_pre_hook(lambda m, args: widths.append(args[0].shape[1])))
            full(torch.randn(1, 1, 256))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(widths, [256, 128, 64])

    def test_validation_and_state_roundtrip(self):
        for cls in (FullFrequencyUNet, TruncatedFrequencyUNet):
            for kwargs in ({"n_mels": 255}, {"n_mels": 0}, {"channels": (32, 64)},
                           {"channels": (32, 0, 128)}, {"n_mels": True}):
                with self.assertRaises(ValueError):
                    cls(**kwargs)
            model = cls().eval()
            for shape in ((1, 2, 256), (1, 256), (0, 1, 256), (1, 1, 128)):
                with self.assertRaises(ValueError):
                    model(torch.zeros(shape))
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    model(torch.full((1, 1, 256), value))
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 1, 256, dtype=torch.long))
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = cls().eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            inputs = torch.randn(1, 1, 256)
            self.assertTrue(torch.equal(model(inputs), restored(inputs)))
            custom = cls(n_mels=128, channels=(4, 8, 16))
            self.assertEqual(custom(torch.randn(1, 1, 128)).shape, (1, 6, 26))
        for bins in (0, -1, True, 65):
            with self.assertRaises(ValueError):
                TruncatedFrequencyUNet(pool_bins=bins)
        self.assertEqual(TruncatedFrequencyUNet(pool_bins=1).heads[0].in_features, 224)

    def test_hooks_restore_mixed_modes_and_buffers(self):
        for cls in (FullFrequencyUNet, TruncatedFrequencyUNet):
            model = cls().train()
            model.encoder[0][1].eval()
            modes = [m.training for m in model.modules()]
            state = {key: value.clone() for key, value in model.state_dict().items()}
            hooks = [len(m._forward_hooks) for m in model.modules()]
            inputs = torch.randn(2, 1, 256, requires_grad=True)
            for _ in range(2):
                maps = extract_feature_maps(model, inputs)
                self.assertEqual([m.shape for m in maps.values()], [(2, 32, 256), (2, 128, 64)])
                self.assertTrue(all(m.device.type == "cpu" and not m.requires_grad for m in maps.values()))
                self.assertEqual(modes, [m.training for m in model.modules()])
                self.assertEqual(hooks, [len(m._forward_hooks) for m in model.modules()])
            maps["encoder.0"].zero_()
            fresh = extract_feature_maps(model, inputs)
            self.assertGreater(float(fresh["encoder.0"].sum()), 0)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(state[key], value), key)
            for kwargs, tensor in [({}, torch.randn(1, 2, 256)),
                                   ({"layer_names": ["missing"]}, inputs),
                                   ({"layer_names": ["heads.0"]}, inputs),
                                   ({"layer_names": ["encoder.0", "encoder.0"]}, inputs)]:
                with self.assertRaises(ValueError):
                    extract_feature_maps(model, tensor, **kwargs)
                self.assertEqual(modes, [m.training for m in model.modules()])
                self.assertEqual(hooks, [len(m._forward_hooks) for m in model.modules()])

    def test_plotting_and_failures(self):
        frame = torch.randn(1, 256)
        fig, ax = plot_spectrogram_frame(frame)
        self.assertEqual(len(ax.lines[0].get_xdata()), 256)
        self.assertIs(plot_spectrogram_frame(frame[0], ax=ax)[0], fig)
        maps = extract_feature_maps(FullFrequencyUNet(), frame.unsqueeze(0))
        fig, axes = plot_feature_maps(maps)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(fig.axes), 4)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        self.assertTrue(buffer.getvalue().startswith(b"\x89PNG"))
        for bad in (torch.zeros(2, 256), torch.zeros(255), torch.full((256,), float("nan"))):
            with self.assertRaises(ValueError):
                plot_spectrogram_frame(bad)
        for bad, index in [({}, 0), (maps, 2), (maps, -1), ({"bad": torch.zeros(2, 3)}, 0)]:
            with self.assertRaises(ValueError):
                plot_feature_maps(bad, sample_index=index)

    def test_training_integration(self):
        fixture = test_infrastructure.Acceptance()
        fixture.setUp()
        try:
            train, val = fixture.data()
            for cls in (FullFrequencyUNet, TruncatedFrequencyUNet):
                model = cls(channels=(4, 8, 16))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                checkpoint = fixture.root / cls.__name__
                history = train_model(model, optimizer, train, val, num_epochs=1, batch_size=8,
                                      device="cpu", checkpoint_dir=checkpoint)
                self.assertEqual(len(history), 1)
                saved = {key: value.clone() for key, value in model.state_dict().items()}
                self.assertEqual(train_model(model, optimizer, train, val, num_epochs=1, batch_size=8,
                                            device="cpu", checkpoint_dir=checkpoint), history)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(saved[key], value))
        finally:
            fixture.tearDown()

    def test_additive_shapes_and_exact_sum(self):
        model = AdditiveFrequencyEncoder().eval()
        inputs = torch.randn(2, 1, 256)
        names = ["pools.0", "pools.1", "encoder.2", "proj1_to_3", "proj2_to_3"]
        maps = extract_feature_maps(model, inputs, layer_names=names)
        self.assertEqual([tuple(m.shape) for m in maps.values()],
                         [(2, 64, 128), (2, 128, 64), (2, 256, 64), (2, 256, 64), (2, 256, 64)])
        with torch.no_grad():
            fused = maps["encoder.2"] + maps["proj1_to_3"] + maps["proj2_to_3"]
            expected = torch.stack([head(fused.flatten(1)) for head in model.heads], dim=1)
            self.assertTrue(torch.equal(model(inputs), expected))
        self.assertEqual(len(model.pools), 2)
        self.assertFalse(any(isinstance(m, (nn.ConvTranspose1d, nn.AdaptiveAvgPool1d)) for m in model.modules()))
        self.assertEqual(model.heads[0].in_features, 16384)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2730140)
        self.assertEqual(sum(head.weight.numel() for head in model.heads), 2555904)
        self.assertEqual(sum(p.numel() for module in (model.proj1_to_3, model.proj2_to_3)
                             for p in module.parameters()), 49664)
        modes = [m.training for m in model.modules()]
        state = {key: value.clone() for key, value in model.state_dict().items()}
        default_maps = extract_feature_maps(model, inputs)
        self.assertEqual([tuple(m.shape) for m in default_maps.values()], [(2, 64, 256), (2, 256, 64)])
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(state[key], value) for key, value in model.state_dict().items()))

    def test_additive_backward_validation_and_roundtrip(self):
        for batch in (1, 3):
            model = AdditiveFrequencyEncoder()
            output = model(torch.randn(batch, 1, 256))
            self.assertEqual(output.shape, (batch, 6, 26))
            self.assertTrue(torch.isfinite(output).all())
            GuitarTabLoss()(output, torch.randint(0, 26, (batch, 6))).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for n_mels in (0, -4, 255, True, 256.0):
            with self.assertRaises(ValueError):
                AdditiveFrequencyEncoder(n_mels)
        model = AdditiveFrequencyEncoder().eval()
        for bad in [None, torch.zeros(0, 1, 256), torch.zeros(1, 2, 256), torch.zeros(1, 256),
                    torch.zeros(1, 1, 128), torch.zeros(1, 1, 256, dtype=torch.long),
                    torch.full((1, 1, 256), float("nan")), torch.full((1, 1, 256), float("inf"))]:
            with self.assertRaises(ValueError):
                model(bad)
        for n_mels in (4, 132):
            custom = AdditiveFrequencyEncoder(n_mels).eval()
            self.assertEqual(custom(torch.randn(1, 1, n_mels)).shape, (1, 6, 26))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = AdditiveFrequencyEncoder().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        inputs = torch.randn(1, 1, 256)
        self.assertTrue(torch.equal(model(inputs), restored(inputs)))

    def test_additive_training_integration(self):
        fixture = test_infrastructure.Acceptance()
        fixture.setUp()
        try:
            train, val = fixture.data()
            model = AdditiveFrequencyEncoder()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            checkpoint = fixture.root / "additive"
            history = train_model(model, optimizer, train, val, num_epochs=1, batch_size=8,
                                  device="cpu", checkpoint_dir=checkpoint)
            self.assertEqual(len(history), 1)
            saved = {key: value.clone() for key, value in model.state_dict().items()}
            self.assertEqual(train_model(model, optimizer, train, val, num_epochs=1, batch_size=8,
                                        device="cpu", checkpoint_dir=checkpoint), history)
            self.assertTrue(all(torch.equal(saved[key], value) for key, value in model.state_dict().items()))
        finally:
            fixture.tearDown()

    def test_baseline_structure_and_sequential_forward(self):
        model = TabCNNBaseline().eval()
        expected_types = [[nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Dropout],
                          [nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Dropout],
                          [nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d, nn.ReLU, nn.Dropout]]
        for block, types in zip(model.encoder, expected_types):
            self.assertEqual([type(layer) for layer in block], types)
            self.assertEqual(block[0].kernel_size, (3,))
            self.assertEqual(block[0].padding, (1,))
            self.assertEqual(block[-1].p, 0.25)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2587548)
        self.assertEqual(model.heads[0].in_features, 16384)
        inputs = torch.randn(2, 1, 256)
        maps = extract_feature_maps(model, inputs, layer_names=["encoder.0", "encoder.1", "encoder.2"])
        self.assertEqual([tuple(m.shape) for m in maps.values()],
                         [(2, 32, 256), (2, 64, 256), (2, 128, 128)])
        with torch.no_grad():
            features = inputs
            for block in model.encoder:
                features = block(features)
            expected = torch.stack([head(features.flatten(1)) for head in model.heads], dim=1)
            self.assertTrue(torch.equal(model(inputs), expected))
            self.assertTrue(torch.equal(model(inputs), model(inputs)))
        dropout_input = torch.ones(2, 32, 256)
        for block in model.encoder:
            self.assertTrue(torch.equal(block[-1](dropout_input), dropout_input))
            block[-1].train()
            self.assertTrue((block[-1](dropout_input) == 0).any())

    def test_baseline_validation_state_and_hooks(self):
        for width in (0, -2, 255, True, 256.0):
            with self.assertRaises(ValueError):
                TabCNNBaseline(width)
        model = TabCNNBaseline().train()
        model.encoder[0][1].eval()
        modes = [m.training for m in model.modules()]
        state = {k: v.clone() for k, v in model.state_dict().items()}
        inputs = torch.randn(1, 1, 256)
        maps = extract_feature_maps(model, inputs)
        self.assertEqual([tuple(m.shape) for m in maps.values()], [(1, 32, 256), (1, 128, 128)])
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in model.modules()))
        for bad in [None, torch.zeros(0, 1, 256), torch.zeros(1, 2, 256), torch.zeros(1, 256),
                    torch.zeros(1, 1, 128), torch.zeros(1, 1, 256, dtype=torch.long),
                    torch.full((1, 1, 256), float("nan")), torch.full((1, 1, 256), float("inf"))]:
            with self.assertRaises(ValueError):
                model(bad)
        for width in (2, 130):
            self.assertEqual(TabCNNBaseline(width).eval()(torch.randn(1, 1, width)).shape, (1, 6, 26))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = TabCNNBaseline().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(model.eval()(inputs), restored(inputs)))

    def test_all_default_models_forward_backward(self):
        for cls in (FullFrequencyUNet, TruncatedFrequencyUNet, AdditiveFrequencyEncoder, TabCNNBaseline):
            for batch in (1, 3):
                with self.subTest(model=cls.__name__, batch=batch):
                    model = cls()
                    logits = model(torch.randn(batch, 1, 256))
                    self.assertEqual(logits.shape, (batch, 6, 26))
                    self.assertTrue(torch.isfinite(logits).all())
                    GuitarTabLoss()(logits, torch.randint(0, 26, (batch, 6))).backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    for head in model.heads:
                        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_all_default_models_exact_mid_epoch_resume(self):
        fixture = test_infrastructure.Acceptance()
        fixture.setUp()

        def assert_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_equal(a, b)
            else:
                self.assertEqual(left, right)

        def interrupt(epoch, batch, step, loss):
            if step == 1:
                raise KeyboardInterrupt("Simulated interruption after saved step")

        try:
            train, val = fixture.data()
            for cls in (FullFrequencyUNet, TruncatedFrequencyUNet, AdditiveFrequencyEncoder, TabCNNBaseline):
                with self.subTest(model=cls.__name__):
                    torch.manual_seed(123)
                    full = cls()
                    full_optimizer = torch.optim.Adam(full.parameters(), lr=1e-3)
                    options = dict(num_epochs=2, batch_size=8, device="cpu", seed=42)
                    full_dir = fixture.root / (cls.__name__ + "-full")
                    resumed_dir = fixture.root / (cls.__name__ + "-resumed")
                    expected = train_model(full, full_optimizer, train, val, checkpoint_dir=full_dir, **options)
                    torch.manual_seed(123)
                    broken = cls()
                    broken_optimizer = torch.optim.Adam(broken.parameters(), lr=1e-3)
                    with self.assertRaises(KeyboardInterrupt):
                        train_model(broken, broken_optimizer, train, val, checkpoint_dir=resumed_dir,
                                    on_batch=interrupt, **options)
                    cursor = torch.load(resumed_dir / "last_checkpoint.pt", weights_only=True)
                    self.assertEqual(cursor["epoch"], 0)
                    self.assertEqual(cursor["next_batch"], 1)
                    self.assertEqual(cursor["global_step"], 1)
                    resumed = cls()
                    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3)
                    actual = train_model(resumed, resumed_optimizer, train, val,
                                         checkpoint_dir=resumed_dir, **options)
                    assert_equal(expected, actual)
                    assert_equal(full.state_dict(), resumed.state_dict())
                    assert_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
                    full_checkpoint = torch.load(full_dir / "last_checkpoint.pt", weights_only=True)
                    resumed_checkpoint = torch.load(resumed_dir / "last_checkpoint.pt", weights_only=True)
                    assert_equal(full_checkpoint, resumed_checkpoint)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
