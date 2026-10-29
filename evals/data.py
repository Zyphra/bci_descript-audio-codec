"""Load dataset manifests, cached tokens, or native codec outputs using fixed splits."""
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def load_tokens(dataset, model=None, *, cache=None, eeg_root=None,
                device="cuda", batch_size=32, target_std=0.3, clip=1.0):
    """Return in-memory CPU IDs, frozen codebook, labels, and train/val/test indices."""
    metadata = Path(dataset) / "cache.json"
    data = json.loads(Path(metadata).read_text())
    signal = data["signal_metadata"]
    assert signal["signal_unit"] == "V" and signal["processing_stage"] == "before_model_preprocessing"
    records = {r["recording_id"]: r for r in data["recordings"]}
    split_by = data.get("split_by", "subject")
    assert split_by in ("subject", "trial")
    rows = {w["example_id"]: w if split_by == "trial" else records[w["recording_id"]] for w in data["windows"]}
    if split_by == "subject":
        subjects = [{r["subject"] for r in records.values() if r["split"] == s}
                    for s in ("train", "val", "test")]
        assert all(subjects[i].isdisjoint(subjects[j]) for i in range(3) for j in range(i))
    windows = sorted(data["windows"], key=lambda w: w["example_id"])
    wanted = torch.tensor([w["example_id"] for w in windows])
    labels = torch.tensor([w.get("label", records[w["recording_id"]].get("label")) for w in windows])
    splits = {s: torch.tensor([i for i, w in enumerate(windows)
                              if rows[w["example_id"]]["split"] == s]) for s in ("train", "val", "test")}
    chunks, examples = [], []
    if cache is not None:
        cache = Path(cache)
        manifest = json.loads((cache / "metadata.json").read_text())
        assert manifest["dataset_sha256"] == sha256(metadata), "Dataset differs from token cache"
        assert manifest["channel_names"] == signal["channel_names"], "Channel order differs"
        for name, digest in manifest["files"].items():
            assert sha256(cache / name) == digest, f"Corrupt token cache: {name}"
            value = torch.load(cache / name, map_location="cpu", weights_only=True)
            if name == "codebook.pt":
                codebook = value
            else:
                chunks.append(value["ids"])
                examples.append(value["example_ids"])
    else:
        from audiotools import AudioSignal
        from audiotools.data.transforms import EEGNormalize

        frequency = signal["frequency_hz"]
        assert model.sample_rate == frequency and model.n_codebooks == 1
        model.to(device).eval().requires_grad_(False)
        codebook = model.quantizer.quantizers[0].codebook.weight.detach().cpu().clone()
        normalize = EEGNormalize(target_std=target_std, clip=clip)
        for recording_id, row in records.items():
            selected = [w for w in windows if w["recording_id"] == recording_id]
            if not selected:
                continue
            path = Path(eeg_root or dataset) / row["signal_file"]
            assert sha256(path) == row["signal_sha256"], f"Corrupt EEG cache: {path}"
            recording = torch.load(path, map_location="cpu", weights_only=True)
            eeg = recording["eeg"]
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                x = torch.stack([eeg[:, w["start_sample"]:w["stop_sample"]] for w in batch])
                x = x.to(device=device, dtype=torch.float32).flatten(0, 1).unsqueeze(1)
                x = normalize._transform(AudioSignal(x, sample_rate=frequency)).audio_data
                _, ids, _, _, _ = model.encode(model.preprocess(x, frequency))
                chunks.append(ids.reshape(len(batch), len(signal["channel_names"]), -1).cpu())
                examples.append(torch.tensor([w["example_id"] for w in batch]))
    examples, order = torch.cat(examples).sort()
    ids = torch.cat(chunks)[order]
    assert len(examples.unique()) == len(examples), "Duplicate token examples"
    indices = torch.searchsorted(examples, wanted)
    assert indices.max() < len(examples) and torch.equal(examples[indices], wanted), "Missing token examples"
    ids = ids[indices]
    return ids, codebook, labels, splits
