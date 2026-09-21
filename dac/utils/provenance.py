"""Local, per-launch provenance. Does not read credentials or dump the environment."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE, timeout=30
    )


def snapshot_code(package_path, destination):
    """Record Git state and archive current source, including untracked source.

    The archive is the working tree, not HEAD. Generated runs/data, editor files,
    and environment files are excluded. Non-Git installations still get a source
    snapshot and an explicit unavailable Git status.
    """
    package_path = Path(package_path).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True)
    info = {"package_path": str(package_path)}
    try:
        root = Path(_git(package_path, "rev-parse", "--show-toplevel").decode().strip())
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        info.update(
            root=str(root),
            commit=_git(root, "rev-parse", "HEAD").decode().strip(),
            branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
            dirty=bool(status.strip()),
        )
        (destination / "git_status.txt").write_bytes(status)
        (destination / "working_tree.patch").write_bytes(
            _git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
        )
        names = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        candidates = [root / name.decode() for name in names.split(b"\0") if name]
    except (OSError, subprocess.SubprocessError) as error:
        root = package_path
        info.update(root=str(root), commit=None, branch=None, dirty=None,
                    git_error=type(error).__name__)
        candidates = list(root.rglob("*"))

    excluded = {".git", ".venv", "venv", "__pycache__", "runs", "ckpt", "analysis",
                "build", "dist", "node_modules", ".vscode"}
    extensions = {".py", ".yaml", ".yml", ".toml", ".cfg", ".sh"}
    files = {}
    archive = destination / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(set(candidates)):
            relative = path.relative_to(root)
            if excluded.intersection(relative.parts) or not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in extensions and not path.name.startswith("requirements"):
                continue
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            tar.add(path, arcname=str(relative), recursive=False)
    (destination / "source_files.json").write_text(json.dumps(files, indent=2) + "\n")
    info.update(source_file_count=len(files),
                source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    return info


def create_launch_record(output, packages):
    """Create a unique record before model/data initialization can fail."""
    now = datetime.now(timezone.utc)
    launch_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
    launch = Path(output) / "launches" / launch_id
    launch.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("torch", "numpy", "argbind", "wandb", "mne", "descript-audiotools",
                 "descript-audio-codec"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "schema_version": 1, "launch_id": launch_id,
        "started_at_utc": now.isoformat(), "cwd": str(Path.cwd()),
        "argv": list(sys.argv), "python_executable": sys.executable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": versions,
        "code": {name: snapshot_code(path, launch / "code" / name)
                 for name, path in packages.items()},
    }
    (launch / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    return launch


def link_wandb_launch(launch, run):
    """Link the local source/config record to its W&B run without uploading code."""
    path = Path(launch) / "launch.json"
    record = json.loads(path.read_text())
    record["wandb"] = {"id": run.id, "url": run.url}
    path.write_text(json.dumps(record, indent=2) + "\n")
    run.config.update({"provenance": {
        "launch_id": record["launch_id"], "directory": str(Path(launch).resolve()),
        "code": record["code"],
    }}, allow_val_change=True)
