"""Add six mean curves to an existing authenticated W&B workspace.

Requires wandb-workspaces. Reads both current mean_<metric> keys and legacy
<metric>_mean keys, so existing runs can be compared with future launches.
"""

import argparse


def add_mean_panels(workspace):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    for split in ("train", "val"):
        name = f"{split}_codebooks"
        section = next((s for s in workspace.sections if s.name == name), None)
        if section is None:
            section = ws.Section(name=name, is_open=True)
            workspace.sections.append(section)
        for metric in ("utilization", "perplexity", "dominant_fraction"):
            keys = [f"{name}/mean_{metric}", f"{name}/{metric}_mean"]
            existing = next((p for p in section.panels
                             if isinstance(p, wr.LinePlot) and p.y
                             and set(p.y).issubset(set(keys))), None)
            if existing is None:
                existing = wr.LinePlot()
                section.panels.append(existing)
            existing.title = f"mean_{metric}"
            existing.x = "training_step"
            existing.y = keys
            existing.log_y = metric == "dominant_fraction"
            existing.aggregate = False
        section.is_open = True
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-url", required=True,
                        help="Project workspace URL including its nw query, not a run URL")
    parser.add_argument("--dry-run", action="store_true", help="Read and preview without saving")
    args = parser.parse_args()
    import wandb_workspaces.workspaces as ws

    workspace = add_mean_panels(ws.Workspace.from_url(args.workspace_url))
    for section in workspace.sections:
        if section.name in ("train_codebooks", "val_codebooks"):
            for panel in section.panels:
                if (getattr(panel, "title", None) or "").startswith("mean_"):
                    print(f"{section.name}/{panel.title}: {panel.y}")
    if not args.dry_run:
        workspace.save()
        print(workspace.url)


if __name__ == "__main__":
    main()
