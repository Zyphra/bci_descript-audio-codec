# One Slurm job running several experiment configs on a single GPU.
# Submitted by submit_cw_sweep.py; not meant to be run by hand.
#
# Edit codebook_sizes, loss_combinations, dataset_names, run_cntr, and
# jobs_per_GPU in submit_cw_sweep.py, then from the repo root:
#   Preview:  /data/groups/bci/chris/workspace/venv_dac/bin/python scripts/submit_cw_sweep.py --dry-run
#   Submit:   /data/groups/bci/chris/workspace/venv_dac/bin/python scripts/submit_cw_sweep.py
#
# Follow an experiment:
#   tail -f runs/slurm_logs/crt_cw_026_<jobid>.log
#
# Usage: sbatch scripts/train_cw_slurm.sh CONFIG.yml [CONFIG.yml ...]
# Each config sets save_path and WandB.name.

"""Edit the lists below, then run with --dry-run to preview or without it to submit."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import subprocess
import yaml

REPO_DIR = Path(__file__).resolve().parents[1]


run_name_base = "nameo" # "msstft"
run_cntr_start = 0 #6
#
jobs_per_GPU = 3
num_workers = 16 # 16 - turn down if dataloading bottleneck.

#
notes_tag = " " # default:  " "

# Parameters to sweep. Each loss tuple is one combination, crossed with every size.
codebook_sizes = [512, 2048, 8192] # [512, 1024, 2048, 4096, 8192]
#
loss_combinations = [
  #                                             adv/           vq/ 
  # mel   wave   |  stft   mag   log   phi   |  feat  gen   |  commit code
  # 15.0  15.0      15.0   1.0   1.0   1.0      2.0   1.0      0.25   1.0 - nonzero defaults
#   (  0.0, 15.0,     15.0,  1.0,  0.0,  0.0,     0.0,  0.0,     0.25,  1.0 , "wav+stft-(mag)+vq" ),
#   (  0.0, 15.0,     15.0,  1.0,  0.0,  0.0,     2.0,  1.0,     0.25,  1.0 , "wav+stft-(mag)+adv(No MPD)+vq" ),
  (  0.0,  0.0,     15.0,  1.0,  0.0,  1.0,     0.0,  0.0,     0.25,  1.0 , "stft-(mag,phi)+vq" ),
#   (  0.0,  0.0,     15.0,  1.0,  0.0,  1.0,     2.0,  1.0,     0.25,  1.0 , "stft-(mag,phi)+adv(No MPD)+vq" ),
]
#
MSSTFT_window_lengths =[ 
    ([16, 32, 64, 128, 256], "Smaller STFT wins"),
    ([64, 256], "Default STFT wins"),
]

#
dataset_names = ["rand1k"] ## ["alice", "rand1k", "rand10k"]
dataset_paths = { # Note: must be a dict with keys matching dataset_names.
  "alice": "/data/groups/bci/datasets/alice/Alice/cache/prep2",
  "rand1k": "/data/groups/bci/datasets/processed/v8_sets/1k",
  "rand10k": "/data/groups/bci/datasets/processed/v8_sets/10k",
}


def build_experiments(base_config):
    """
    Build a list of experiments, each with a unique configuration.
    Everything not defined here comes from YAML.
    """
    experiments = []
    next_run_cntr = run_cntr_start
    for dataset_name in dataset_names:
        for loss_tuple in loss_combinations:
            for STFT_wins in MSSTFT_window_lengths:
                for codebook_size in codebook_sizes:
                    #
                    # Copy before modifying, so experiments never change one another.
                    config = deepcopy(base_config)
                    #
                    # Number of workers
                    config["num_workers"] = num_workers
                    #
                    # Codebook size
                    config["DAC.codebook_size"] = codebook_size
                    #
                    # Losses
                    config.setdefault("lambdas", {})["mel/loss"] = loss_tuple[0]
                    config.setdefault("lambdas", {})["waveform/loss"] = loss_tuple[1]
                    config.setdefault("lambdas", {})["STFT/loss"] = loss_tuple[2]
                    config["MultiScaleSTFTLoss.mag_weight"] = loss_tuple[3]
                    config["MultiScaleSTFTLoss.log_weight"] = loss_tuple[4]
                    config["MultiScaleSTFTLoss.phase_weight"] = loss_tuple[5]
                    config.setdefault("lambdas", {})["adv/feat_loss"] = loss_tuple[6]
                    config.setdefault("lambdas", {})["adv/gen_loss"] = loss_tuple[7]
                    config.setdefault("lambdas", {})["vq/commitment_loss"] = loss_tuple[8]
                    config.setdefault("lambdas", {})["vq/codebook_loss"] = loss_tuple[9]
                    #
                    # STFT window lengths
                    config["MultiScaleSTFTLoss.window_lengths"] = STFT_wins[0]
                    #
                    # Replace inherited datasets with this sweep's selected dataset.
                    config["train/build_dataset.folders"] = {dataset_name: [dataset_paths[dataset_name]+'/train']}
                    config["val/build_dataset.folders"] = {dataset_name: [dataset_paths[dataset_name]+'/val']}
                    #
                    # Name
                    name = f"{run_name_base}_{next_run_cntr:03d}"
                    config["WandB.name"] = name
                    config["save_path"] = f"runs/{name}"
                    next_run_cntr += 1
                    #
                    # Notes : configure this to show short hand version of the config
                    config["WandB.notes"] = f"TBD:{notes_tag}codes={config['DAC.n_codebooks']}x{codebook_size}. {loss_tuple[10]}. {STFT_wins[1]}. {dataset_name} data."
                    #
                    experiments.append((name, config))
    return experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_DIR / "conf/base_cw_eeg.yml")
    parser.add_argument("--dry-run", action="store_true", help="Preview groups without writing files or submitting jobs")
    args = parser.parse_args()
    if jobs_per_GPU < 1:
        parser.error("jobs_per_GPU must be at least 1")
    experiments = build_experiments(yaml.safe_load(args.config.read_text()))
    groups = [experiments[i:i + jobs_per_GPU] for i in range(0, len(experiments), jobs_per_GPU)]
    print(f"{len(experiments)} experiments, {len(groups)} Slurm jobs, at most {jobs_per_GPU} experiments per GPU", flush=True)
    for index, group in enumerate(groups, 1):
        print(f"GPU job {index}: {', '.join(name for name, _ in group)}", flush=True)
    if args.dry_run or not experiments:
        return

    # Save a separate config for every run; nested lambdas cannot be overridden
    # using --lambdas.waveform/loss in this version of ArgBind.
    sweep_dir = REPO_DIR / "runs/sweeps" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sweep_dir.mkdir(parents=True)
    (REPO_DIR / "runs/slurm_logs").mkdir(parents=True, exist_ok=True)
    config_groups = []
    for group in groups:
        paths = []
        for name, config in group:
            path = sweep_dir / f"{name}.yml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            paths.append(str(path))
        config_groups.append(paths)
    print(f"Experiment configs: {sweep_dir}", flush=True)
    for paths in config_groups:
        subprocess.run(
            ["sbatch", str(REPO_DIR / "scripts/train_cw_slurm.sh"), *paths],
            cwd=REPO_DIR, check=True,
        )


if __name__ == "__main__":
    main()
