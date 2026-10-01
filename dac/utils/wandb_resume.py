"""Persist W&B identity alongside an experiment's checkpoints."""
import json
from pathlib import Path
import uuid


def prepare_wandb_run(output, *, project, entity, resume):
    """Reuse an identity only for an explicitly resumed training run.

    Older experiments can recover their most recent linked launch. Display
    names are deliberately not used as identity: W&B permits duplicate names.
    """
    output = Path(output)
    path = output / "wandb_run.json"
    identity = None
    if resume and path.exists():
        identity = json.loads(path.read_text())
        if identity["project"] != project or identity["entity"] != entity:
            raise ValueError(f"W&B project/entity differs from {path}; use a new save_path")
        if not isinstance(identity.get("id"), str) or not identity["id"]:
            raise ValueError(f"Missing W&B run ID in {path}")
    elif resume:
        for launch in sorted(output.glob("launches/*/launch.json"), reverse=True):
            linked = json.loads(launch.read_text()).get("wandb", {})
            parts = (linked.get("url") or "").rstrip("/").split("/")
            if (linked.get("id") and len(parts) >= 4
                    and parts[-2] == "runs" and parts[-3] == project
                    and (not entity or parts[-4] == entity)):
                identity = {"id": linked["id"], "project": project, "entity": entity}
                break
    if identity is None:
        identity = {"id": uuid.uuid4().hex, "project": project, "entity": entity}
    output.mkdir(parents=True, exist_ok=True)
    # Save before contacting W&B, so an interruption during init keeps the ID.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n")
    temporary.replace(path)
    return identity["id"]
