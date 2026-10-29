"""Submit isolated Slurm evaluations; rerun a snapshot with python -m evals.classify DIRECTORY."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import warnings

from .data import load_tokens, sha256


def save_json(path, value):
    """Publish metrics, baseline scores, and plot history only after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def submit(model, step, output, args, wandb_run, every, sbatch_args, datasets, python=sys.executable,
           models=("transformer", "bag_of_tokens"), token_cache="/data/groups/bci/token_cache", *, config):
    """Freeze weights before submission; singleton jobs serialize this run's evaluations."""
    if len(set(models)) != len(models) or set(models) - {"transformer", "bag_of_tokens"}:
        raise ValueError("classifier_models must contain transformer and/or bag_of_tokens, without duplicates")
    if not models or not every or step % every:
        return
    output = Path(output).resolve()
    destination = output / f"step_{step:08d}"
    output.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        job = {**config, "step": step, "architecture": type(model).__name__,
               "models": list(models), "token_cache": str(Path(token_cache).resolve()),
               "datasets": {name: {"directory": str(Path(path).resolve()),
                                    "sha256": sha256(Path(path) / "cache.json")} for name, path in datasets.items()},
               "target_std": args.get("EEGNormalize.target_std", 0.3),
               "clip": args.get("EEGNormalize.clip", 1.0), "wandb": None}
        if wandb_run is not None:
            selection = hashlib.sha256(json.dumps([sorted(models), config], sort_keys=True).encode()).hexdigest()[:8]
            job["wandb"] = {"project": wandb_run.project, "entity": wandb_run.entity,
                            "id": f"{wandb_run.id}-classifiers-{selection}", "name": wandb_run.name + "-classifiers",
                            "group": wandb_run.group or wandb_run.id,
                            "mode": args.get("WandB.mode", "online")}
            job["parent_run"] = wandb_run.url
        with TemporaryDirectory(prefix=".snapshot-", dir=output) as scratch:
            staging = Path(scratch) / "data"
            staging.mkdir()
            model.save(staging / "weights.pth", package=False)
            save_json(staging / "job.json", job)
            staging.rename(destination)
    if (destination / "job_id.txt").exists():
        return
    # Do not export the training job's GPU assignment, Slurm options, or DDP identity.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))
           and k not in ("CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                         "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run_id = hashlib.sha256(str(output.parent.parent).encode()).hexdigest()[:12]
    command = ["sbatch", *sbatch_args, "--parsable", "--dependency=singleton", "--export=ALL",
               f"--job-name=classifier-{run_id}",
               f"--chdir={Path(__file__).resolve().parents[1]}",
               f"--output={destination}/slurm-%j.log", "--wrap",
               shlex.join([python, "-m", "evals.classify", str(destination)])]
    try:
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=15)
        (destination / "job_id.txt").write_text(result.stdout.strip() + "\n")
        print(f"Classifier step {step}: submitted Slurm job {result.stdout.strip()}", flush=True)
    except (OSError, subprocess.SubprocessError) as error:
        warnings.warn(f"Classifier submission failed; snapshot retained at {destination}: "
                      f"{getattr(error, 'stderr', None) or error}", RuntimeWarning)


def fit(ids, codebook, labels, splits, job, device):
    """Choose weights by validation accuracy, then measure all splits at those weights."""
    from functools import partial
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models.bag_of_tokens import BagOfTokensClassifier
    from .models.transformer_classifier import TransformerClassifier, train, test

    results = {}
    n_classes = int(labels.max()) + 1
    for name in job["models"]:
        torch.manual_seed(job["seed"])
        if name == "transformer":
            model = torch.nn.Sequential(torch.nn.Embedding.from_pretrained(codebook, freeze=True),
                TransformerClassifier(n_classes, channels=ids.shape[1], coarse_time=ids.shape[2],
                                      codebook_dim=codebook.shape[1], **job.get("transformer", {})))
            loss = torch.nn.functional.cross_entropy
        else:
            model = BagOfTokensClassifier(len(codebook), n_classes=n_classes, **job.get("bag_of_tokens", {}))
            loss = partial(model.loss, n_train=len(splits["train"]))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=job["learning_rate"])
        loaders = {s: DataLoader(TensorDataset(ids[i], labels[i]), batch_size=job["batch_size"],
                   shuffle=s == "train", generator=torch.Generator().manual_seed(job["seed"]))
                   for s, i in splits.items()}
        best = -1
        for epoch in range(1, job["epochs"] + 1):
            training = train(model, loaders["train"], optimizer, loss_function=loss)
            validation = test(model, loaders["val"])
            assert not training["missing_gradients"]
            if validation["balanced_accuracy"] > best:
                best = validation["balanced_accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                results[name] = {"epoch": epoch, "val": validation}
            print(f"{name} epoch {epoch}: val_balanced_accuracy={validation['balanced_accuracy']:.4f}", flush=True)
        model.load_state_dict(best_state)
        results[name].update({s: test(model, loaders[s]) for s in ("train", "test")})
    return results


def baseline_scores(dataset, job, output, device):
    """Fit each fixed tokenizer/classifier once per dataset and evaluation configuration."""
    scores = {}
    protocol = [job[k] for k in ("seed", "epochs", "batch_size", "learning_rate")]
    protocol += [sha256(p) for p in sorted(Path(__file__).parent.rglob("*.py"))]
    for source in ("tfm", "labram"):
        manifests = list((Path(job["token_cache"]) / dataset / source).glob("*/metadata.json"))
        if len(manifests) != 1:
            raise ValueError(f"Expected one {source} cache for {dataset} under {job['token_cache']}")
        manifest, data = manifests[0], None
        scores[source] = {}
        for model in job["models"]:
            identity = [sha256(manifest), job["datasets"][dataset]["sha256"], model, job.get(model, {}), *protocol]
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            path = output / "baselines" / f"{dataset}-{source}-{model}-{key}.json"
            if not path.exists():
                if data is None:
                    data = load_tokens(job["datasets"][dataset]["directory"], cache=manifest.parent)
                metrics = fit(*data, {**job, "models": [model]}, device)[model]
                save_json(path, metrics)
            scores[source][model] = json.loads(path.read_text())
    return scores


def evaluate(directory, device="cuda"):
    import torch
    from dac.model import DAC, k8MOSS

    directory = Path(directory)
    job = json.loads((directory / "job.json").read_text())
    if not job["models"]:
        return {}
    assert job["epochs"] > 0 and job["batch_size"] > 0
    assert job["datasets"], "No evaluation datasets configured"
    for dataset in job["datasets"].values():
        assert sha256(Path(dataset["directory"]) / "cache.json") == dataset["sha256"], "Evaluation dataset changed"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    result_path = directory / "metrics.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else None
    if results is None or any("test" not in score for models in results["datasets"].values()
                              for score in models.values()):
        blob = torch.load(directory / "weights.pth", map_location="cpu", weights_only=False)
        model = {"DAC": DAC, "k8MOSS": k8MOSS}[job["architecture"]](**blob["metadata"]["kwargs"])
        model.load_state_dict(blob["state_dict"], strict=True)
        results = {"tokenizer_step": job["step"], "checkpoint_sha256": sha256(directory / "weights.pth"),
                   "datasets": {}, "baselines": {}}
        for name, dataset in job["datasets"].items():
            data = load_tokens(dataset["directory"], model, device=device, batch_size=job["batch_size"],
                               target_std=job["target_std"], clip=job["clip"])
            model.cpu()
            results["datasets"][name] = fit(*data, job, device)
            del data  # Current-tokenizer IDs stay in RAM and are discarded after fitting.
            results["baselines"][name] = baseline_scores(name, job, directory.parents[1], device)
        del model, blob
        save_json(result_path, results)
    if job["wandb"] is not None:
        import wandb

        history_path = directory.parents[1] / f"history-{job['wandb']['id']}.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else {}
        history[str(job["step"])] = results["datasets"]
        save_json(history_path, history)
        with wandb.init(**job["wandb"], resume="allow", dir=str(directory), job_type="classifier",
                        config={"parent_run": job["parent_run"], "datasets": job["datasets"],
                                "models": job["models"], "classifier_config": {
                                    k: job.get(k, {}) for k in ("seed", "epochs", "batch_size", "learning_rate",
                                                               "transformer", "bag_of_tokens")}}) as run:
            run.define_metric("tokenizer_step", hidden=True)
            panels = {"tokenizer_step": job["step"]}
            for dataset in job["datasets"]:
                for model in job["models"]:
                    steps = sorted(int(s) for s, row in history.items()
                                   if "test" in row.get(dataset, {}).get(model, {}))
                    series = {"tokenizer": [history[str(s)][dataset][model] for s in steps]}
                    series.update({source: [results["baselines"][dataset][source.lower()][model]] * len(steps)
                                   for source in ("TFM", "LaBraM")})
                    for metric in ("cross_entropy", "balanced_accuracy"):
                        ys = [[row[split][metric] for row in scores]
                              for scores in series.values() for split in ("train", "val", "test")]
                        panels[f"classifier/{dataset}/{model}/{metric}"] = wandb.plot.line_series(
                            steps, ys, keys=[f"{source}/{split}" for source in series
                                            for split in ("train", "val", "test")],
                            title=f"{dataset}: {model} {metric.replace('_', ' ')}", xname="Tokenizer step")
            run.log(panels)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda")
    evaluate(**vars(parser.parse_args()))
