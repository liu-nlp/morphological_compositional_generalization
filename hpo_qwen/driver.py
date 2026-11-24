# driver.py
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna

# --- Config ---
MODEL_ID = "Qwen/Qwen3-30B-A3B-Base" #"Qwen/Qwen3.5-35B-A3B-Base"
TRAIN_DATA_PATH = [PATH_TO_TRAIN_DATA]
VAL_DATA_PATH = [PATH_TO_VAL_DATA]
SPLIT = "productivity_Estonian_test1"
OUTPUT_DIR = [PATH_TO_OUTPUT_DIR]
SAVE_STRATEGY = "epoch"
NUM_EPOCHS = 10
BATCH_SIZE = 8
GRAD_ACCUM = 1
LOGGING_STEPS = 50

HPO_ROOT = Path([PATH_TO_HPO_FOLDER])
SCRIPTS_ROOT = Path([PATH_TO_TRAIN_SCRIPTS])
SBATCH_PATH = SCRIPTS_ROOT / "trial.sh"
SBATCH_FINAL_PATH = SCRIPTS_ROOT / "final.sh"
STORAGE = f"sqlite:///{HPO_ROOT}/study.db"
STUDY_NAME = MODEL_ID + "_" + SPLIT


def objective(trial):
    '''Starts a single optuna trial for hyperparameter search, returning the validation loss.'''
    # --- Sample hyperparameters ---
    lora_r = trial.suggest_categorical("lora_r", [16, 32, 64, 128])
    alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [1, 2])
    lora_alpha = lora_r * alpha_multiplier
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    if MODEL_ID in ["google/gemma-4-26B-A4B", "Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3.5-35B-A3B-Base"]:
        dropout = 0.0
    else:
        dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    # --- Per-trial paths ---
    result_file = HPO_ROOT / "results" / f"trial_{trial.number}.json"
    
    print(f"\n=== Trial {trial.number} ===", flush=True)
    print(f"  r={lora_r} alpha={lora_alpha} lr={lr:.2e} dropout={dropout:.3f} weight_decay={weight_decay:.2e}", flush=True)

    cmd = [
        "sbatch", "--wait", "--parsable", 
        str(SBATCH_PATH),
        str(MODEL_ID), 
        str(TRAIN_DATA_PATH), 
        str(VAL_DATA_PATH),
        str(BATCH_SIZE), 
        str(GRAD_ACCUM), 
        str(lora_r), 
        str(lora_alpha), 
        str(lr), 
        str(dropout), 
        str(weight_decay),
        str(trial.number), 
        str(result_file),
#        str(VAL_DATA_PATH_OTHER)
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        print(f"sbatch invocation crashed: {e!r}", flush=True)
        return float("inf")

    job_id = (proc.stdout or "").strip().split(";")[0]
    print(f"  SLURM job: {job_id}, sbatch exit={proc.returncode}", flush=True)

    if proc.returncode != 0:
        print(f"  trial failed (sbatch exit {proc.returncode})", flush=True)
        print(f"  stderr: {proc.stderr[-500:]}", flush=True)
        return float("inf")

    # --- Read the result file ---
    if not result_file.exists():
        print(f"  trial finished but no result file at {result_file}", flush=True)
        return float("inf")

    try:
        data = json.loads(result_file.read_text())
        eval_loss = float(data["eval_loss"])
    except Exception as e:
        print(f"  failed to parse result file: {e!r}", flush=True)
        return float("inf")

    print(f"  eval_loss = {eval_loss:.4f} (status: {data.get('status', '?')})",
          flush=True)
    return eval_loss


def run_final_training(study):
    '''Takes the best hyperparameters found by optuna and starts the final run of the model.'''
    best = study.best_params
    lora_r = best["lora_r"]
    lora_alpha = lora_r * best["alpha_multiplier"]
    lr = best["learning_rate"]
    if MODEL_ID in ["google/gemma-4-26B-A4B", "Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3.5-35B-A3B-Base"]:
        dropout = 0.0
    else:
        dropout = best["lora_dropout"]
    weight_decay = best["weight_decay"]

    #OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Final training ===", flush=True)
    print(f"  best eval_loss from HPO: {study.best_value:.4f}", flush=True)
    print(f"  best params: {best}", flush=True)
    print(f"  num_epochs: {NUM_EPOCHS}", flush=True)
    print(f"  output:  {OUTPUT_DIR}", flush=True)

    cmd = [
        "sbatch", "--wait", "--parsable", str(SBATCH_FINAL_PATH),
        str(MODEL_ID),
        str(TRAIN_DATA_PATH),
        str(VAL_DATA_PATH),
        str(OUTPUT_DIR),
        str(BATCH_SIZE),
        str(GRAD_ACCUM),
        str(lr), 
        str(weight_decay),
        str(lora_r), 
        str(lora_alpha),
        str(dropout),
        str(LOGGING_STEPS),
        str(NUM_EPOCHS),
        str(SAVE_STRATEGY),
#        str(VAL_DATA_PATH_OTHER)
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    job_id = (proc.stdout or "").strip().split(";")[0]
    print(f"  SLURM job: {job_id}, sbatch exit={proc.returncode}", flush=True)

    if proc.returncode != 0:
        print(f"  FINAL TRAINING FAILED", flush=True)
        return None

    print(f"  Final model + checkpoints in {OUTPUT_DIR}", flush=True)
    return OUTPUT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=5,
                   help="Total trials to run in this driver invocation.")
    p.add_argument("--run-final", action="store_true",
                   help="After HPO, run one full training with the best params.")
    p.add_argument("--skip-hpo", action="store_true",
                   help="Skip HPO and go straight to final training "
                        "(uses existing study's best_params).")
    args = p.parse_args()

    HPO_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("results", "runs", "logs", "final_run"):
        (HPO_ROOT / sub).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        storage=STORAGE,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    if not args.skip_hpo:
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Study: {STUDY_NAME}")
        print(f"  total trials so far: {len(study.trials)}")
        print(f"  completed: {len(completed)}")
        if completed:
            print(f"  best so far: {study.best_value:.4f} -> {study.best_params}")
        print(f"  running {args.n_trials} more trials")
        sys.stdout.flush()

        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== HPO complete ===")
    print(f"Best eval_loss: {study.best_value:.4f}")
    print(f"Best params:    {study.best_params}")

    if args.run_final:
        run_final_training(study)


if __name__ == "__main__":
    main()