#!/bin/bash
# batch_forecast_curves.sh
set -euo pipefail

# ---------------------------
# Usage info
# ---------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 MARKET TRANSFORMER K_SUPPLY K_DEMAND [N_PARALLEL] [N_THREADS]"
    echo
    echo "Runs all forecast combinations in parallel."
    echo "  MARKET      : market name (required)"
    echo "  TRANSFORMER : space-separated list of transformers (required)"
    echo "  K_SUPPLY    : space-separated list of K_supply values (required)"
    echo "  K_DEMAND    : space-separated list of K_demand values (required)"
    echo "  N_PARALLEL  : number of parallel jobs (default: 8)"
    echo "  N_THREADS   : OMP threads per job     (default: 1)"
    echo
    echo "Tip: set N_PARALLEL * N_THREADS <= total logical cores."
    echo "Example: $0 GME \"fpca zst\" \"2 3 4 5\" \"10 15\" 48 2"
    exit 0
fi

# ---------------------------
# Arguments
# ---------------------------
MARKET=${1:?          "Error: MARKET (arg 1) is required"}
TRANSFORMER_ARG=${2:? "Error: TRANSFORMER (arg 2) is required"}
K_SUPPLY_ARG=${3:?    "Error: K_SUPPLY (arg 3) is required"}
K_DEMAND_ARG=${4:?    "Error: K_DEMAND (arg 4) is required"}
N_PARALLEL=${5:-8}
N_THREADS=${6:-1}

# Convert space-separated strings to arrays
read -ra TRANSFORMER <<< "$TRANSFORMER_ARG"
read -ra K_SUPPLY <<< "$K_SUPPLY_ARG"
read -ra K_DEMAND <<< "$K_DEMAND_ARG"

# ---------------------------
# Activate Python environment
# ---------------------------
source ../.venvs/moc_forecast/bin/activate

# ---------------------------
# Define static parameters
# ---------------------------
export START_DATE=20240101
export END_DATE=20241231
export PROB_START_DATE=20240701
export LAGS_ENDOG="1237"
export LAGS_EXOG="017"
export EXOG_STRUCTURE="concurrent"
export TRANS_TYPE="static"
export WINDOW_LENGTH=364
export IC="aic"
export NSIM=1000
export ENDOG_PATH=data/processed/"$MARKET"/sdts.pkl
export EXOG_PATH=data/processed/"$MARKET"/predictors.csv
export SAVE_FOLDER=data/output/"$MARKET"/curve_based/

export N_THREADS   # Export so run_one can read it
export MARKET # Export so run_one can read it

# ---------------------------
# Define parameter arrays
# ---------------------------
# Convert space-separated strings to arrays
CHOICE_K=("none")
AUTOCORR_STRUCTURE=("concurrent" "full")
CROSSCORR_STRUCTURE=("none" "concurrent")
CALIB_WINDOW=(28 56 91 182)

# ---------------------------
# Function executed by each parallel job
# ---------------------------
run_one() {
    line="$1"
    read -r Ks Kd chK ac cc trans cw <<< "$line"

    # Set threading for this job only
    export OMP_NUM_THREADS="$N_THREADS"
    export MKL_NUM_THREADS="$N_THREADS"
    export NUMEXPR_NUM_THREADS="$N_THREADS"
    export OPENBLAS_NUM_THREADS="$N_THREADS"

    RUN_NAME="${ac:0:4}_${cc:0:4}_${LAGS_ENDOG}_${EXOG_STRUCTURE:0:4}_${LAGS_EXOG}_${trans}_${TRANS_TYPE}_${Ks}_${Kd}_${chK}_${WINDOW_LENGTH}_${IC}_${START_DATE}_${END_DATE}"
    FORECASTER_PATH="data/output/${MARKET}/curve_based/forecasters/${RUN_NAME}.pkl"
    OUTPUT_PATH="data/output/${MARKET}/curve_based/price_quantiles/${RUN_NAME}_${cw}D.pkl"

    python -m scripts.curve_based.probabilistic_forecast \
        --forecaster_path "$FORECASTER_PATH" \
        --output_path "$OUTPUT_PATH" \
        --calib_window "$cw" \
        --start_date "$PROB_START_DATE" \
        --n_sim "$NSIM" \
        --n_jobs 1 \
        --correct_monotonicity
}

export -f run_one

# ---------------------------
# Function to generate all parameter combinations
# ---------------------------
gen_combinations() {
    for Ks in "${K_SUPPLY[@]}"; do
        for Kd in "${K_DEMAND[@]}"; do
            for chK in "${CHOICE_K[@]}"; do
                for ac in "${AUTOCORR_STRUCTURE[@]}"; do
                    for cc in "${CROSSCORR_STRUCTURE[@]}"; do
                        for trans in "${TRANSFORMER[@]}"; do
                            for cw in "${CALIB_WINDOW[@]}"; do
                                echo "$Ks $Kd $chK $ac $cc $trans $cw"
                            done
                        done
                    done
                done
            done
        done
    done
}

# ---------------------------
# Sanity check
# ---------------------------
TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "?")
echo "[INFO] Logical cores available : $TOTAL_CORES"
echo "[INFO] N_PARALLEL=$N_PARALLEL | N_THREADS=$N_THREADS | effective cores used: $((N_PARALLEL * N_THREADS))"
if [[ "$TOTAL_CORES" != "?" && $((N_PARALLEL * N_THREADS)) -gt "$TOTAL_CORES" ]]; then
    echo "[WARN] N_PARALLEL * N_THREADS exceeds available cores — expect oversubscription."
fi

# ---------------------------
# Run all combinations in parallel
# ---------------------------
echo "[INFO] Running with N_PARALLEL=$N_PARALLEL, N_THREADS=$N_THREADS ..."
gen_combinations | xargs -P "$N_PARALLEL" -I {} bash -c 'run_one "{}"'