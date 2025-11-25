#!/bin/bash
# batch_forecast_curves.sh
set -euo pipefail

# ---------------------------
# Usage info
# ---------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [N_PARALLEL]"
    echo
    echo "Runs all forecast combinations in parallel."
    echo "If N_PARALLEL is not provided, defaults to 8."
    exit 0
fi

# ---------------------------
# HPC / threading settings
# ---------------------------
export OMP_NUM_THREADS=1        # Limit threading per Python process
N_PARALLEL=${1:-8}              # Number of parallel jobs (default = 8)

# ---------------------------
# Activate Python environment
# ---------------------------
source ~/Projects/.venvs/moc_forecast/bin/activate

# ---------------------------
# Define static parameters
# ---------------------------
export START_DATE=20240101
export END_DATE=20241231
export ENDOG_PATH=data/processed/sdts.pkl
export EXOG_PATH=data/source/predictors.pkl
export SAVE_FOLDER=data/output/

# ---------------------------
# Define parameter arrays
# ---------------------------
# K_SUPPLY=(5 8 13)
# K_DEMAND=(1 3 4)
# AR_STRUCTURE=("concurrent" "full")
# VAR_STRUCTURE=("none" "concurrent")
# TRANSFORMER=("fpca" "zst")
K_SUPPLY=(0)
K_DEMAND=(0)
CHOICE_K=("threshold-elbow" "threshold" "elbow")
AR_STRUCTURE=("concurrent" "full")
VAR_STRUCTURE=("none" "concurrent")
TRANSFORMER=("fpca")

# ---------------------------
# Function executed by each parallel job
# ---------------------------
run_one() {
    line="$1"
    read -r Ks Kd chK ar var trans <<< "$line"

    python -m scripts.forecast_curves \
        --endog_path "$ENDOG_PATH" \
        --exog_path "$EXOG_PATH" \
        --save_folder "$SAVE_FOLDER" \
        --start_date "$START_DATE" \
        --end_date "$END_DATE" \
        --K_supply "$Ks" \
        --K_demand "$Kd" \
        --choice_K "$chK" \
        --transformer "$trans" \
        --ar_structure "$ar" \
        --var_structure "$var"
}

export -f run_one

# ---------------------------
# Function to generate all parameter combinations
# ---------------------------
gen_combinations() {
    for Ks in "${K_SUPPLY[@]}"; do
        for Kd in "${K_DEMAND[@]}"; do
            for chK in "${CHOICE_K[@]}"; do
                for ar in "${AR_STRUCTURE[@]}"; do
                    for var in "${VAR_STRUCTURE[@]}"; do
                        for trans in "${TRANSFORMER[@]}"; do
                            echo "$Ks $Kd $chK $ar $var $trans"
                        done
                    done
                done
            done
        done
    done
}

# ---------------------------
# Run all combinations in parallel
# ---------------------------
echo "[INFO] Running with N_PARALLEL=$N_PARALLEL ..."
gen_combinations | xargs -P "$N_PARALLEL" -I {} bash -c 'run_one "{}"'