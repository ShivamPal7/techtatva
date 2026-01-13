import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm

INPUT_CSV = 'cleaned_data/dataset_with_paths.csv'
OUTPUT_CSV = 'cleaned_data/dataset_verified.csv'

def check_file(row):
    """
    Checks if the file in the row exists and is valid (non-empty).
    Returns (index, is_valid) tuple.
    """
    idx, path = row
    
    # 1. Validation: Check if path is a string
    if not isinstance(path, str):
        return idx, False

    # 2. Check existence
    if not os.path.exists(path):
        return idx, False
        
    # 3. Check 0-byte (Corrupt)
    try:
        if os.path.getsize(path) == 0:
            return idx, False
    except OSError:
        return idx, False

    # (Skipping full PIL Image.open verify for speed as requested, 
    # relying on 0-byte check which caught previous errors)
    return idx, True

def clean_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total entries: {len(df)}")

    # Pre-filter: Drop NaN paths immediately to avoid unnecessary processing
    initial_len = len(df)
    df = df.dropna(subset=['path'])
    print(f"Dropped {initial_len - len(df)} NaN path entries.")

    valid_indices = []
    
    # Prepare data for parallel execution: list of (index, path)
    # We iterate over rows but only need index and path
    rows_to_check = list(zip(df.index, df['path']))

    print(f"Verifying {len(rows_to_check)} images using multi-threading...")
    
    # Use ThreadPoolExecutor for IO-bound task (checking files)
    # Windows handles this well. 
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        # Submit all tasks
        futures = {executor.submit(check_file, row): row for row in rows_to_check}
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows_to_check)):
            idx, is_valid = future.result()
            if is_valid:
                valid_indices.append(idx)

    # Sort indices to maintain original order roughly
    valid_indices.sort()
    
    print(f"\nVerification Complete.")
    print(f"Original Valid Paths: {len(rows_to_check)}")
    print(f"Final Valid Images: {len(valid_indices)}")
    print(f"Removed: {len(rows_to_check) - len(valid_indices)} files (Missing/Empty)")

    if valid_indices:
        df_clean = df.loc[valid_indices]
        df_clean.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved verified dataset to: {OUTPUT_CSV}")
    else:
        print("No valid images found!")

if __name__ == "__main__":
    clean_dataset()
