import numpy as np
import pandas as pd
import h5py

import argparse
import subprocess
import os
from os.path import join


def preprocess_point_forecasts(
    prices_true_path,
    prices_pred_path,
    output_folder
):
    os.makedirs(output_folder, exist_ok=True)

    prices_true = pd.read_csv(prices_true_path, index_col=0, parse_dates=True)
    prices_pred = pd.read_csv(prices_pred_path, index_col=0, parse_dates=True)

    df = pd.DataFrame(index=prices_pred.index)
    df["real"] = prices_true.loc[df.index, "NAT"]
    df["pred"] = prices_pred.loc[:, "NAT"]

    for h in range(24):
        df_h = df[df.index.hour == h].copy()
        df_h.index = df_h.index.strftime("%Y%m%d").astype(int)
        df_h.to_csv(join(output_folder, f"h{h}.csv"))

    print("\n✅ Preprocessing completed")


def run_postforecasts(args):
    cmd = [
        args.julia,
        args.julia_script,
        "--input_folder", join(args.processed_folder, '_input'),
        "--output_folder", args.processed_folder,
        "--calibration_windows", args.calibration_windows,
        "--methods", args.methods,
        "--start", str(args.test_start),
        "--stop", str(args.test_end),
        "--quantiles", str(args.quantiles),
    ]

    print("\n🚀 Running Julia postforecasts...\n")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    print("\n✅ Julia postforecasts completed\n")
    

def postprocess_postforecasts(
    input_folder,
    output_folder,
    calibration_windows,
    methods
):
    calibration_windows = [int(w) for w in calibration_windows.split(",")]
    methods = methods.split(",")

    os.makedirs(output_folder, exist_ok=True)

    # Load quantile levels once
    sample = os.path.join(
        input_folder,
        f"{calibration_windows[0]}D",
        methods[0],
        "h1.quantf"
    )

    with h5py.File(sample, "r") as f:
        prob_levels = f["prob"][:]

    for window in calibration_windows:
        for method in methods:
            input_dir = os.path.join(input_folder, f"{window}D", method)
            hourly = []

            for h in range(1, 25):
                path = os.path.join(input_dir, f"h{h}.quantf")
                if not os.path.exists(path):
                    continue

                with h5py.File(path, "r") as f:
                    data = f["pred"][:].T
                    ids = f["id"][:].astype(str)

                hour = f"{h-1:02d}"
                idx = pd.to_datetime(
                    [f"{d}{hour}" for d in ids],
                    format="%Y%m%d%H"
                )

                hourly.append(pd.DataFrame(data, index=idx, columns=prob_levels))

            if hourly:
                final = pd.concat(hourly).sort_index()
                # /!\ postprocessing specific to GME /!\
                final.clip(lower=0, inplace=True)
                out = os.path.join(output_folder, f"{method}_{window}D.pkl")
                final.to_pickle(out)
                print(f"✅ Saved {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full postforecast pipeline")

    # --- Shared / core parameters ---
    parser.add_argument("--prices_true_path", type=str, required=True)
    parser.add_argument("--prices_pred_path", type=str, required=True)

    parser.add_argument("--processed_folder", type=str, required=True,
                        help="Julia processed folder")

    parser.add_argument("--output_folder", type=str, required=True,
                        help="Final output quantiles folder")
    
    parser.add_argument("--test_start", type=int, required=True, help="Test start date in YYYYMMDD format")
    parser.add_argument("--test_end", type=int, required=True, help="Test end date in YYYYMMDD format")

    # --- Julia parameters ---
    parser.add_argument("--calibration_windows", type=str, required=True,
                        help="Comma-separated list, e.g. 28,56,91,182")
    
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated list, e.g. normal,cp,idr")
    
    parser.add_argument("--quantiles", type=int, default=99)

    # --- Julia executable ---
    parser.add_argument("--julia", type=str, default="julia",
                        help="Path to Julia executable")

    parser.add_argument("--julia_script", type=str, default="scripts/postforecasts.jl")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Step 1 – preprocessing
    preprocess_point_forecasts(
        prices_true_path=args.prices_true_path,
        prices_pred_path=args.prices_pred_path,
        output_folder=join(args.processed_folder, '_input')
    )

    # Step 2 – Julia postforecasts script
    run_postforecasts(args)

    # Step 3 – postprocessing
    postprocess_postforecasts(
        input_folder=args.processed_folder,
        output_folder=args.output_folder,
        calibration_windows=args.calibration_windows,
        methods=args.methods
    )

    print("\n🎉 Full postforecast pipeline completed successfully")

