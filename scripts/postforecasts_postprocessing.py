import h5py
import pandas as pd
import os
import numpy as np

def process_julia_forecasts(
        input_folder="data/processed/postforecasts/Naive",
        output_folder="data/output/price-based/quant/Naive"
    ):
    """
    Loads HDF5 files using the correct keys ('pred' for data, 'id' for index, 
    and 'prob' for quantile levels), aggregates the 24-hour forecasts, 
    and saves them to a dedicated CSV file.

    Args:
        base_output_dir (str): The base directory where the Julia script
                               saved its output (e.g., 'data/output/postforecasts/fARX').
    """

    # --- 1. Define the parameters from the Julia script ---
    calibration_windows = [28, 56, 91, 182]
    methods = ["normal"]
    
    # The dedicated directory for the final CSVs
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving final CSV files to: {output_folder}")
    print("-" * 80)

    # --- 2. Load Quantile Levels (Probabilities) once ---
    # We must load one file to get the correct 'prob' array for column names.
    try:
        sample_file_path = os.path.join(input_folder, f"{calibration_windows[0]}D", methods[0], "h1.quantf")
        if not os.path.exists(sample_file_path):
             print(f"Error: Cannot find sample file to extract quantile levels: {sample_file_path}")
             return
             
        with h5py.File(sample_file_path, 'r') as f:
            # Load the 1D array of quantile levels (e.g., [0.01, 0.02, ..., 0.99])
            prob_levels = f['prob'][:]
            # Create column names: 0.01, 0.02, etc. (formatted to 2 decimal places)
            num_quantiles = len(prob_levels)
            print(f"Successfully loaded {num_quantiles} quantile levels for column names.")
            print(f"Example columns: {prob_levels[0]}, {prob_levels[-1]}")
            
    except Exception as e:
        print(f"Error during initial quantile level loading: {e}")
        return
    print("-" * 80)


    for window in calibration_windows:
        window_dir_name = f"{window}D"
        
        for method in methods:
            print(f"Processing Method: {method.upper()}, Window: {window}D")
            
            input_dir = os.path.join(input_folder, window_dir_name, method)
            all_hourly_dfs = []
            
            # --- 3. Load the 24 hourly HDF5 files ---
            for h in range(1, 25): # h1 to h24
                file_name = f"h{h}.quantf"
                file_path = os.path.join(input_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"  Warning: File not found: {file_path}. Skipping.")
                    continue

                try:
                    with h5py.File(file_path, 'r') as f:
                        # Use 'pred' for the quantile forecasts (the values)
                        quantile_data = f['pred'][:].T
                        
                        # Use 'id' for the timestamps/index
                        date_ints = f['id'][:]
                        date_strings = date_ints.astype(str)
                        
                        # Format the hour with a leading zero (01 to 24)
                        hour_str = f'{h-1:02d}'
                        
                        # Combine date and hour, e.g., '2024070101'
                        datetime_strings = [f'{d}{hour_str}' for d in date_strings]

                        # 4. Convert to DatetimeIndex, specifying the input format
                        # Format: %Y (Year) %m (Month) %d (Day) %H (Hour)
                        index = pd.to_datetime(datetime_strings, format='%Y%m%d%H')

                        if quantile_data.shape[1] != num_quantiles:
                            print(f"  Error: Expected {num_quantiles} quantiles, but found {quantile_data.shape[1]} in {file_name}. Skipping.")
                            continue
                            
                        df_h = pd.DataFrame(
                            quantile_data, 
                            index=index, 
                            columns=prob_levels
                        )
                        
                        all_hourly_dfs.append(df_h)
                        
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
            
            
            # --- 4. Concatenate and Finalize Data ---
            if all_hourly_dfs:
                final_df = pd.concat(all_hourly_dfs).sort_index()

                # Set the index format to 'YYYY-MM-DD HH:MM:SS' as requested
                # final_df.index = final_df.index.strftime('%Y-%m-%d %H:%M:%S')

                # --- 5. Save to CSV ---
                output_filename = f"{method}_{window}D.pkl"
                output_filepath = os.path.join(output_folder, output_filename)
                
                final_df.to_pickle(output_filepath)
                print(f"  ✅ Successfully created and saved: {output_filename}")
            else:
                print(f"  ⚠️ Could not load any data for method {method} with window {window}D.")
        print("-" * 80)

if __name__ == "__main__":
    process_julia_forecasts()