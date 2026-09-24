"""Publish two reusable presets from a logged-in shell using W&B >= 0.30.

This setup script does not modify runs, workspaces, or training configuration.
It prints the YAML settings to enable the shared-run panels on future launches.
"""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="W&B user/team owning the runs")
    parser.add_argument("--export-dir", type=Path, help="Write Vega JSON locally without authentication")
    args = parser.parse_args()
    # Avoid importing the training/torch stack into a one-time setup environment.
    spec = importlib.util.spec_from_file_location(
        "codebook_charts", Path(__file__).resolve().parents[1] / "dac/utils/codebook_charts.py"
    )
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for scale in charts.PRESET_NAMES:
            path = args.export_dir / f"codebook_mean_sem_{scale}.json"
            path.write_text(json.dumps(charts.mean_sem_spec(scale == "log"), indent=2) + "\n")
            print(path)
        return

    import wandb
    if not hasattr(wandb.Api, "create_custom_chart"):
        parser.error("This setup command requires wandb>=0.30. Use a separate setup environment; "
                     "the training environment can keep its existing W&B version.")
    api = wandb.Api()
    presets = {}
    for scale, name in charts.PRESET_NAMES.items():
        presets[scale] = api.create_custom_chart(
            entity=args.entity, name=name, display_name=f"Codebook layer mean ± SEM ({scale})",
            spec_type="vega2", access="private", spec=charts.mean_sem_spec(scale == "log"),
        )
    print("\nAdd to your training YAML:")
    print("WandB.codebook_chart_presets:")
    for scale, preset in presets.items():
        print(f"  {scale}: {preset}")


if __name__ == "__main__":
    main()
