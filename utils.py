import os
import json
import time
from pathlib import Path
from viz import plot_results

def save_run(config, history, run_name=None):
    # Create a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = f"run_{timestamp}_{config.get('optimizer_name', 'adamw')}"
    run_dir = Path(f"logs/{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    # Prepare data to save
    # Note: We can't save the 'model' object or 'dataset' objects in JSON easily.
    # So we filter the config to just simple types (int, float, str).

    clean_config = {k: v for k, v in config.items() if isinstance(v, (int, float, str))}

    log_data = {
    "config": clean_config,
    "history": history
    }

    with open(f"{run_dir}/{run_name}.json", "w") as f:
        json.dump(log_data, f, indent=4)
    plot_results(history, plot_dir=run_dir)
    # print(f"Run saved to {run_dir}/{run_name}.json")