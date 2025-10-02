import os
import json
import pandas as pd

def recover_unfinished_run(dir):
    all_rows = []

    for i in range(500):
        job_dir = os.path.join(dir, f"zjob_{i}")
        # Load hyper_config.json
        hyper_config_path = os.path.join(job_dir, "hyper_config.json")
        if not os.path.exists(hyper_config_path):
            continue
        with open(hyper_config_path, "r") as f:
            dct = json.load(f)
        # Load tune_results.csv
        tune_df = pd.read_csv(os.path.join(job_dir, "tune_results.csv"))
        # Find row with largest val_acc
        best_row = tune_df.loc[tune_df["val_acc"].idxmax()].to_dict()
        # Extend with dct
        best_row.update(dct)

        # INSERT_YOUR_CODE
        # Read log.txt and extract batch_size from line 30
        log_path = os.path.join(job_dir, "log.txt")
        batch_size_dict = {}
        if os.path.exists(log_path):
            with open(log_path, "r") as logf:
                lines = logf.readlines()
                line_30 = lines[29]  
                # Extract the number after 'batch_size:'
                if "batch_size:" in line_30:
                    batch_size_str = line_30.split("batch_size:")[1].strip()
                    batch_size_val = int(batch_size_str)
                    best_row["batch_size"] = batch_size_val
                else:
                    raise ValueError(f"batch_size not found in log.txt for job {job_dir}")

                line_36 = lines[35]  
                # Extract the number after 'beta:'
                if "beta:" in line_36:
                    beta_str = line_36.split("beta:")[1].strip()
                    beta_val = int(beta_str)
                    best_row["beta"] = beta_val
                else:
                    print(line_36)
                    raise ValueError(f"beta not found in log.txt for job {job_dir}")

        all_rows.append(best_row)

    df = pd.DataFrame(all_rows)
    return df