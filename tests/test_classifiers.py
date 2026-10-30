"""CPU regression checks for isolated, step-indexed dataset evaluation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from dac.model import k8MOSS
from evals import classify
from evals.data import load_tokens, sha256


class ClassifierTests(unittest.TestCase):
    def test_startup_panels_before_results_and_resume(self):
        from wandb.sdk.wandb_summary import Summary

        run = MagicMock()
        summary = {}
        run.summary = Summary(lambda: summary)
        datasets = {"depression": "/missing", "ssvep": "/also-missing"}
        classify.initialize_panels(run, datasets, ["transformer"])
        panels = run.log.call_args.args[0]
        self.assertEqual(set(panels), {f"classifier/{d}_transformer_{m}"
            for d in datasets for m in ("cross_entropy", "balanced_accuracy")})
        self.assertTrue(all(p._log_type == "image-file" for p in panels.values()))
        self.assertTrue(all("Pending evaluation" in p._caption for p in panels.values()))
        self.assertNotIn("tokenizer_step", panels)  # No fabricated evaluation step or scores.
        summary.update(panels)
        run.reset_mock()
        classify.initialize_panels(run, datasets, ["transformer"])
        run.log.assert_not_called()  # Preserve curves on resume.
        summary.clear()
        classify.initialize_panels(run, datasets, [])
        run.log.assert_not_called()

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(33)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.metadata = self.root / "cache.json"
        self.output = self.root / "run/evals/launch"
        self.datasets = {"fixture": str(self.root)}
        self.config = dict(seed=33, epochs=1, batch_size=2, learning_rate=1e-3,
                           transformer=dict(model_dim=16, heads=2, depth=1), bag_of_tokens=dict(C=1.0))
        records, windows = [], []
        for i, split in enumerate(("train", "train", "val", "val", "test", "test")):
            path = self.root / f"{i:03d}.pt"
            torch.save({"eeg": torch.randn(2, 1280, dtype=torch.float64),
                        "example_ids": torch.tensor([i])}, path)
            records.append(dict(recording_id=i, subject=str(i), label=i % 2, split=split,
                                n_samples=1280, signal_file=path.name, signal_sha256=sha256(path)))
            windows.append(dict(example_id=i, recording_id=i, start_sample=0, stop_sample=1280))
        self.metadata.write_text(json.dumps(dict(schema_version=2, recordings=records, windows=windows,
            label_mapping={"a": 0, "b": 1},
            signal_metadata=dict(frequency_hz=256, signal_unit="V", processing_stage="before_model_preprocessing",
                                 channel_names=["C3", "C4"]))))
        self.model = k8MOSS(coarse_time=1, embed_dim=8, projection_stages=[1.0], num_heads=2,
                            n_codebooks=1, codebook_size=8, codebook_dim=2)
        self.token_cache = self.root / "references"
        for dataset in ("fixture", "second"):
            for source in ("tfm", "labram"):
                cache = self.token_cache / dataset / source / "checkpoint"
                cache.mkdir(parents=True)
                torch.save(torch.randn(8, 2), cache / "codebook.pt")
                torch.save(dict(ids=torch.randint(8, (6, 2, 5)), example_ids=torch.arange(6)), cache / "000.pt")
                (cache / "metadata.json").write_text(json.dumps(dict(
                    dataset_sha256=sha256(self.metadata), channel_names=["C3", "C4"],
                    files={p.name: sha256(p) for p in cache.iterdir()})))

    def test_schedule_and_immutable_snapshot(self):
        original = self.model.encoder.patch_projection.bias.detach().clone()
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="7", RANK="3", SBATCH_GPUS="8"), \
                patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="123\n")) as launch:
            for step in range(5):
                classify.submit(self.model, step, self.output, {}, None, 2, ["--gpus=1"], self.datasets, config=self.config, gpu="slurm")
                with torch.no_grad():
                    self.model.encoder.patch_projection.bias.add_(1)
            self.assertEqual(launch.call_count, 3)
            call = launch.call_args
            self.assertIn("--dependency=singleton", call.args[0])
            self.assertTrue(all(k not in call.kwargs["env"] for k in ("CUDA_VISIBLE_DEVICES", "RANK", "SBATCH_GPUS")))
            self.assertTrue(self.model.training)
            self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
            for step in (0, 2, 4):
                path = self.output / f"step_{step:08d}"
                blob = torch.load(path / "weights.pth", weights_only=False)
                torch.testing.assert_close(blob["state_dict"]["encoder.patch_projection.bias"], original + step)
                self.assertEqual(json.loads((path / "job.json").read_text())["step"], step)
            classify.submit(self.model, 0, self.output, {}, None, 2, [], self.datasets, config=self.config, gpu="slurm")
            classify.submit(self.model, 0, self.root / "disabled", {}, None, 0, [], self.datasets, config=self.config, gpu="slurm")
            self.assertEqual(launch.call_count, 3)
            self.assertFalse((self.root / "disabled").exists())
            job_name = next(a for a in call.args[0] if a.startswith("--job-name="))
            classify.submit(self.model, 0, self.output.with_name("resumed"), {}, None, 2, [], self.datasets, config=self.config, gpu="slurm")
            self.assertIn(job_name, launch.call_args.args[0])
            classify.submit(self.model, 0, self.root / "other/evals/launch", {}, None, 2, [], self.datasets,
                            "/eval/python", config=self.config, gpu="slurm")
            self.assertNotIn(job_name, launch.call_args.args[0])
            self.assertTrue(launch.call_args.args[0][-1].startswith("/eval/python "))

    def test_submission_failure_retains_retryable_snapshot(self):
        with patch.object(classify.subprocess, "run", side_effect=subprocess.TimeoutExpired("sbatch", 15)):
            with self.assertWarns(RuntimeWarning):
                classify.submit(self.model, 0, self.output, {}, None, 1, [], self.datasets, config=self.config, gpu="slurm")
        path = self.output / "step_00000000"
        digest = sha256(path / "weights.pth")
        self.assertFalse((path / "job_id.txt").exists())
        with patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="124\n")):
            classify.submit(self.model, 0, self.output, {}, None, 1, [], self.datasets, config=self.config, gpu="slurm")
        self.assertEqual(sha256(path / "weights.pth"), digest)
        self.assertEqual((path / "job_id.txt").read_text().strip(), "124")

    def test_native_cache_parity_and_corruption(self):
        native = load_tokens(self.root, self.model, device="cpu")
        ids, codebook, labels, splits = native
        self.assertEqual(ids.shape, (6, 2, 5))
        torch.testing.assert_close(codebook, self.model.quantizer.quantizers[0].codebook.weight.detach(),
                                   rtol=0, atol=0)
        cache = self.root / "tokens"
        cache.mkdir()
        torch.save(codebook, cache / "codebook.pt")
        torch.save({"ids": ids.flip(0), "example_ids": torch.arange(6).flip(0)}, cache / "000.pt")
        manifest = dict(dataset_sha256=sha256(self.metadata), channel_names=["C3", "C4"],
                        files={p.name: sha256(p) for p in cache.iterdir()})
        (cache / "metadata.json").write_text(json.dumps(manifest))
        cached = load_tokens(self.root, cache=cache)
        for expected, actual in zip(native[:3], cached[:3]):
            torch.testing.assert_close(expected, actual)
        self.assertEqual({s: i.tolist() for s, i in splits.items()},
                         {"train": [0, 1], "val": [2, 3], "test": [4, 5]})
        (cache / "000.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(AssertionError, "Corrupt token cache"):
            load_tokens(self.root, cache=cache)

    def test_end_to_end_and_wandb_step(self):
        from evals.models.transformer_classifier import TransformerClassifier

        config = {**self.config, "seed": 17, "learning_rate": 0.002,
                  "transformer": {**self.config["transformer"], "temporal_pos": "rotary"}, "bag_of_tokens": {"C": 0.5}}
        parent = SimpleNamespace(project="test", entity="test", id="parent", name="tokenizer",
                                 group="", url="https://example.com/parent")
        with patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="123\n")):
            classify.submit(self.model, 0, self.output, {}, parent, 1, [],
                            {**self.datasets, "second": str(self.root)}, token_cache=self.token_cache, config=config, gpu="slurm")
        path = self.output / "step_00000000"
        job = json.loads((path / "job.json").read_text())
        self.assertEqual(job["transformer"]["heads"], 2)
        self.assertEqual(config["transformer"], job["transformer"])
        classify.save_json(path / "metrics.json", {"datasets": {"fixture": {
            "transformer": {"balanced_accuracy": 0.5}}}})  # Old validation-only results need recomputing.
        with patch.dict(os.environ, WANDB_SERVICE="3-0-unix-/tmp/missing-wandb"), \
                patch("wandb.init") as init, patch("wandb.Image", side_effect=lambda figure, **kwargs: figure) as plots, \
                patch("evals.models.transformer_classifier.TransformerClassifier", wraps=TransformerClassifier) as model:
            results = classify.evaluate(path, device="cpu")
            self.assertNotIn("WANDB_SERVICE", os.environ)
            for call in model.call_args_list:
                self.assertEqual(call.kwargs["model_dim"], 16)
                self.assertEqual(call.kwargs["codebook_dim"], 2)
            self.assertEqual(init.call_args.kwargs["id"], parent.id)
            self.assertEqual(init.call_args.kwargs["mode"], "online")
            self.assertNotIn("config", init.call_args.kwargs)  # Preserve the training run's configuration.
            logged = init.return_value.__enter__.return_value.log.call_args.args[0]
            self.assertEqual(logged["tokenizer_step"], 0)
            self.assertEqual(len(logged), 9)  # Step plus two panels per dataset/classifier.
            self.assertIn("classifier/second_transformer_cross_entropy", logged)
            self.assertTrue(all(key.count("/") == 1 for key in logged if key != "tokenizer_step"))
            init.return_value.__enter__.return_value.define_metric.assert_any_call(
                "classifier/*", step_metric="tokenizer_step", summary="none")
        self.assertEqual(set(results["datasets"]), {"fixture", "second"})
        self.assertEqual({p.name for p in path.rglob("*")
                          if p.suffix in (".pt", ".pth", ".npy", ".npz")}, {"weights.pth"})
        self.assertEqual(set(results["datasets"]["fixture"]), {"transformer", "bag_of_tokens"})
        for metrics in results["datasets"]["fixture"].values():
            for split in ("train", "val", "test"):
                self.assertEqual(metrics[split]["n_examples"], 2)
                self.assertEqual(len(metrics[split]["confusion_matrix"]), 2)
        with patch.object(classify, "fit", side_effect=AssertionError("must reuse results")), \
                patch("wandb.init") as init, patch("wandb.Image"):
            self.assertEqual(classify.evaluate(path, device="cpu"), results)
            init.assert_not_called()  # Already published by the active writer.
        # A presentation-only repair must reuse results and serialize as image
        # media, the same type used by reconstruction's automatic panels.
        with patch.object(classify, "fit", side_effect=AssertionError("must reuse results")):
            run = MagicMock()
            classify.publish(run, self.output.parent, force=True)
            repaired = run.log.call_args.args[0]
            images = [value for key, value in repaired.items() if key.startswith("classifier/")]
            self.assertEqual(len(images), 8)
            self.assertTrue(all(value._log_type == "image-file" for value in images))
            self.assertEqual(len(list((self.output.parent / "plots/step_00000000").glob("*.png"))), 8)
            run.reset_mock()
            classify.publish(run, self.output.parent)
            run.log.assert_not_called()
        with patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="124\n")):
            classify.submit(self.model, 2, self.output, {}, parent, 1, [],
                            {**self.datasets, "second": str(self.root)}, token_cache=self.token_cache, config=config, gpu="slurm")
        second = self.output / "step_00000002"
        self.assertEqual(json.loads((second / "job.json").read_text())["wandb"]["id"], job["wandb"]["id"])
        classify.save_json(self.output / "step_00000001/metrics.json",
                           {"tokenizer_step": 1, "datasets": {"fixture": {
                               "transformer": {"balanced_accuracy": 0.5}}}})
        with patch.object(classify, "fit", wraps=classify.fit) as fit, \
                patch("wandb.init") as init, patch("wandb.Image") as plots, \
                patch.object(classify.fcntl, "flock", side_effect=BlockingIOError), \
                patch.object(classify.time, "sleep", side_effect=lambda _: classify.publish(MagicMock(), self.output.parent)):
            later = classify.evaluate(second, device="cpu")
            init.assert_not_called()  # Trainer publishes while holding the run lock.
            self.assertEqual(fit.call_count, 2)  # Only the two current-tokenizer datasets train again.
            for call in plots.call_args_list:
                traces = call.args[0].axes[0].lines
                self.assertEqual(len(traces), 5)  # Tokenizer splits plus test-only references.
                self.assertTrue(all(list(t.get_xdata()) == [0, 2] for t in traces))
                self.assertEqual([t.get_label() for t in traces[:3]], ["tokenizer/train", "tokenizer/val", "tokenizer/test"])
                self.assertEqual([t.get_label() for t in traces[3:]], ["TFM/test", "LaBraM/test"])
                self.assertEqual(len({t.get_color() for t in traces}), 3)
                self.assertEqual([t.get_linestyle() for t in traces], ["-", "--", ":", ":", ":"])
                for reference in traces[3:]:
                    self.assertEqual(reference.get_ydata()[0], reference.get_ydata()[1])
            self.assertEqual([list(t.get_ydata()) for t in plots.call_args_list[0].args[0].axes[0].lines[:3]],
                [[r["datasets"]["fixture"]["transformer"][s]["cross_entropy"] for r in (results, later)]
                 for s in ("train", "val", "test")])
        with patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="125\n")):
            classify.submit(self.model, 4, self.output, {}, parent, 1, [], self.datasets,
                            token_cache=self.token_cache, config={**config, "learning_rate": 0.003}, gpu="slurm")
        changed = json.loads((self.output / "step_00000004/job.json").read_text())
        self.assertEqual(changed["wandb"]["id"], parent.id)

    def test_baseline_config_reuse(self):
        job = {**self.config, "models": ["transformer", "bag_of_tokens"],
               "token_cache": str(self.token_cache),
               "datasets": {"fixture": {"directory": str(self.root), "sha256": sha256(self.metadata)}}}
        with patch.object(classify, "fit", return_value={"transformer": {}, "bag_of_tokens": {}}) as fit:
            classify.baseline_scores("fixture", job, self.output, "cpu")
            classify.baseline_scores("fixture", job, self.output, "cpu")
            self.assertEqual(fit.call_count, 4)
            job["transformer"]["model_dim"] = 128
            classify.baseline_scores("fixture", job, self.output, "cpu")
            self.assertEqual(fit.call_count, 6)  # Only the two transformer baselines changed.
            job["transformer"] = dict(reversed(list(job["transformer"].items())))
            classify.baseline_scores("fixture", job, self.output, "cpu")
            self.assertEqual(fit.call_count, 6)
            job["learning_rate"] = 0.002
            classify.baseline_scores("fixture", job, self.output, "cpu")
            self.assertEqual(fit.call_count, 10)

    def test_window_labels_trial_splits_and_multiclass(self):
        data = json.loads(self.metadata.read_text())
        data.update(split_by="trial", label_mapping={"a": 0, "b": 1, "c": 2})
        path = self.root / "trials.pt"
        torch.save({"eeg": torch.randn(2, 9 * 1024, dtype=torch.float64)}, path)
        data["recordings"] = [dict(recording_id=0, n_samples=9 * 1024, signal_file=path.name,
                                   signal_sha256=sha256(path))]
        data["windows"] = [dict(example_id=i, recording_id=0, start_sample=i * 1024,
                                stop_sample=(i + 1) * 1024, label=i % 3,
                                split=("train", "val", "test")[i // 3]) for i in range(9)]
        self.metadata.write_text(json.dumps(data))
        tokens = load_tokens(self.root, self.model, device="cpu")
        self.assertEqual(tokens[0].shape, (9, 2, 4))
        self.assertEqual(tokens[2].tolist(), [0, 1, 2] * 3)
        metrics = classify.fit(*tokens, dict(seed=33, epochs=1, batch_size=3, learning_rate=1e-3,
                                            models=["transformer", "bag_of_tokens"]), "cpu")
        for result in metrics.values():
            for split in ("train", "val", "test"):
                self.assertEqual(len(result[split]["confusion_matrix"]), 3)

    def test_metrics_use_validation_selected_weights(self):
        from evals.models import transformer_classifier

        observed = []

        def score(model, loader):
            observed.append((model.weight.detach().clone(), loader.dataset.tensors[0].flatten().sort().values.tolist()))
            return {"balanced_accuracy": float(len(observed) == 1)}

        splits = {s: torch.arange(i * 2, i * 2 + 2) for i, s in enumerate(("train", "val", "test"))}
        with patch.object(transformer_classifier, "test", side_effect=score):
            result = classify.fit(torch.arange(6).reshape(6, 1, 1), torch.zeros(6, 2), torch.arange(6) % 2,
                                  splits, dict(seed=33, epochs=2, batch_size=2, learning_rate=0.01,
                                               models=["bag_of_tokens"]), "cpu")["bag_of_tokens"]
        self.assertEqual(result["epoch"], 1)
        self.assertEqual([ids for _, ids in observed], [[2, 3], [2, 3], [0, 1], [4, 5]])
        self.assertFalse(torch.equal(observed[0][0], observed[1][0]))
        for weights, _ in observed[2:]:
            torch.testing.assert_close(weights, observed[0][0], rtol=0, atol=0)

    def test_classifier_selection(self):
        with patch.object(classify.subprocess, "run") as launch:
            classify.submit(self.model, 0, self.output, {}, None, 1, [], self.datasets, models=[], config=self.config, gpu="slurm")
            launch.assert_not_called()
            self.assertFalse(self.output.exists())
        for name in ("transformer", "bag_of_tokens"):
            output = self.root / name / "evals/launch"
            parent = SimpleNamespace(project="test", entity="test", id=name, name=name, group="", url="")
            with patch.object(classify.subprocess, "run", return_value=SimpleNamespace(stdout="123\n")):
                classify.submit(self.model, 0, output, {}, parent, 1, [], self.datasets,
                                models=[name], token_cache=self.token_cache, config=self.config, gpu="slurm")
            path = output / "step_00000000"
            job = json.loads((path / "job.json").read_text())
            job["epochs"] = 1
            classify.save_json(path / "job.json", job)
            with patch("wandb.init") as init, patch("wandb.Image"):
                result = classify.evaluate(path, device="cpu")
                panels = init.return_value.__enter__.return_value.log.call_args.args[0]
                self.assertEqual(set(panels), {"tokenizer_step", f"classifier/fixture_{name}_cross_entropy",
                                              f"classifier/fixture_{name}_balanced_accuracy"})
                self.assertEqual(set(result["datasets"]["fixture"]), {name})

    def test_local_gpu_selection(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-tokenizer", RANK="3"), \
                patch.object(classify.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as launch:
            for step, options, expected in ((0, {}, "GPU-tokenizer"), (2, {"gpu": "5"}, "5")):
                classify.submit(self.model, step, self.output, {}, None, 1, [], self.datasets,
                                config=self.config, **options)
                call = launch.call_args
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], expected)
                self.assertNotIn("RANK", call.kwargs["env"])
                self.assertTrue(call.kwargs["start_new_session"])
                self.assertEqual(call.args[0][:3], ["flock", "-F", str(self.output.parent / "classifier.lock")])
                path = self.output / f"step_{step:08d}"
                self.assertEqual(json.loads((path / "job.json").read_text())["gpu"], expected)
                self.assertEqual((path / "job_id.txt").read_text().strip(), "local:123")
