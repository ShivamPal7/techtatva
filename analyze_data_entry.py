
import pandas as pd
import numpy as np
import re

def analyze_entry(file_path):
    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    log(f"# Analysis Report for {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        log(f"Error reading file: {e}")
        return

    log(f"\n- **Rows**: {df.shape[0]}\n- **Columns**: {df.shape[1]}")
    log(f"- **Column Names**: `{list(df.columns)}`")

    # 1. Header Check
    log("\n## 1. Column Header Check")
    clean_cols = df.columns.str.strip()
    if any(df.columns != clean_cols):
        log("- **Issue**: Headers contain leading/trailing whitespace.")
        log(f"  - Original: `{list(df.columns)}`")
        df.columns = clean_cols
    else:
        log("- Headers look clean.")

    # 2. Column Analysis
    log("\n## 2. Column-by-Column Analysis")

    # Image Index
    col = 'Image Index'
    log(f"\n### Column: `{col}`")
    if col in df.columns:
        nulls = df[col].isnull().sum()
        log(f"- **Missing Values**: {nulls}")
        
        # Check extensions
        non_png = df[~df[col].astype(str).str.lower().str.endswith('.png')]
        if not non_png.empty:
             log(f"- **Issue**: Found {len(non_png)} rows not ending in .png.")
             log("  - Examples:\n```\n" + str(non_png[col].head()) + "\n```")
        
        # Check for duplicates? usually one row per image, but verify unique constraint if expected
        dupes = df[col].duplicated().sum()
        if dupes > 0:
            log(f"- **Note**: Found {dupes} duplicate Image Index entries (one image might have multiple rows or just duplicates?).")

    # Finding Labels
    col = 'finding_labels' # Note: previously identified as valid column name, but check exact casing in file
    # Previous run showed 'finding_labels' key error if case mismatch, let's auto-find
    
    possible_names = [c for c in df.columns if 'finding' in c.lower()]
    if possible_names:
        col = possible_names[0]
        log(f"\n### Column: `{col}`")
        
        # Check numeric garbage
        # distinct labels check
        all_labels = set()
        numeric_garbage = []
        
        for idx, val in df[col].dropna().items():
            val_str = str(val)
            # Check for numeric-only strings like "123"
            if re.match(r'^\d+$', val_str):
                numeric_garbage.append(val_str)
            
            # Split by pipe
            labels = val_str.split('|')
            all_labels.update(labels)
            
        log(f"- **Unique Labels Found**: {len(all_labels)}")
        log(f"- **Vocabulary**: `{sorted(list(all_labels))}`")
        
        if numeric_garbage:
             log(f"- **Critical Issue**: Found {len(numeric_garbage)} numeric-only labels (garbage).")
             log("  - Examples:\n```\n" + str(numeric_garbage[:5]) + "\n```")

    # Patient Age
    col = 'patient_age' # Usually found
    # Auto-find if needed
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    if age_cols:
        col = age_cols[0]
        log(f"\n### Column: `{col}`")
        
        # Check formats
        non_numeric_cnt = 0
        placeholders = []
        range_errors = []
        
        for idx, val in df[col].items():
            val_str = str(val).strip()
            
            # 1. Parsing Check
            parsed_val = None
            try:
                # Try simple float
                parsed_val = float(val_str)
            except:
                # Try parsing "058Y" 
                clean = re.sub(r'[YMD]', '', val_str)
                try:
                    parsed_val = float(clean)
                except:
                    # Try word map
                    pass
            
            if parsed_val is None:
                non_numeric_cnt += 1
                continue
                
            # 2. Logic Check
            if parsed_val < 0:
                range_errors.append(val)
            elif parsed_val > 120 and parsed_val not in [150, 999]: # Assuming 150/999 are placeholders
                range_errors.append(val) # Unrealistic old age not placeholder
            elif parsed_val in [150, 999, 0]: # Common placeholders
                placeholders.append(val)
                
        log(f"- **Format Issues**: {non_numeric_cnt} unparseable entries (text/garbage).")
        if non_numeric_cnt > 0:
             # Sample non-numeric
             non_nums = df[pd.to_numeric(df[col], errors='coerce').isna()][col]
             log("  - Examples:\n```\n" + str(non_nums.head()) + "\n```")
             
        log(f"- **Range Errors (<0 or >120)**: {len(range_errors)}")
        if range_errors: log(f"  - Examples: {range_errors[:5]}")
        
        log(f"- **Placeholders (0, 150, 999)**: {len(placeholders)}")
        if placeholders: log(f"  - Examples: {placeholders[:5]}")

    # Gender
    col = 'Gender'
    if col in df.columns:
        log(f"\n### Column: `{col}`")
        log("- **Value Counts**:\n```\n" + str(df[col].value_counts()) + "\n```")
        # Check for case inconsistencies
        # e.g. M vs Male, F vs Female
        unique_vals = df[col].dropna().unique()
        log(f"- **Unique Strings**: {unique_vals}")

    # View Position
    col = 'View Position'
    if col in df.columns:
        log(f"\n### Column: `{col}`")
        log("- **Value Counts**:\n```\n" + str(df[col].value_counts()) + "\n```")
        # Verify strict list
        valid_views = ['PA', 'AP', 'Lateral'] # Lateral often normalized from LL/RL/Lat
        
        # Check for 'SIDEWAYS' or lower case
        invalids = df[~df[col].isin(valid_views) & df[col].notna()]
        if not invalids.empty:
            log(f"- **Issue**: Found {len(invalids)} non-standard view positions.")
            log("  - Examples:\n```\n" + str(invalids[col].unique()) + "\n```")

    # --- User Specific Checklist Checks ---
    log("\n## 3. User Checklist Verification (Adarsh & Dev Rules)")
    
    # 1. Image Index Specifics
    log("\n### Image Index Rules")
    bad_patterns = ['invalid_image.png', 'IMG_00000117_000.png', '00141_.png', '00000376_011.PNG']
    for pat in bad_patterns:
        cnt = df[df['Image Index'] == pat].shape[0]
        if cnt > 0:
            log(f"- **Found**: `{pat}` (Count: {cnt})")
    
    uppercase_ext = df[df['Image Index'].str.contains(r'\.PNG$', regex=True, na=False)]
    if not uppercase_ext.empty:
        log(f"- **Inconsistent Extension**: Found {len(uppercase_ext)} files ending in `.PNG`")

    # 2. Finding Labels Specifics
    log("\n### Finding Labels Rules")
    bad_labels = ['123', 'XYZ_Disease', 'Unknown_Disorder', 'None']
    vocabulary = set()
    if 'finding_labels' in df.columns:
        for val in df['finding_labels'].dropna():
            vocabulary.update(str(val).split('|'))
    
    for bl in bad_labels:
        if bl in vocabulary:
            log(f"- **Found Invalid Label**: `{bl}`")
            
    # 3. Age Specifics
    log("\n### Patient Age Rules")
    age_col = 'patient_age'
    if age_col in df.columns:
        # Check specific values mentioned
        for val in [0, 999, 150, -5, "Twenty Five", "Unknown", "45.7"]:
            # Need strict string match or numeric match
            # "45.7" might be float 45.7.
            try:
                if isinstance(val, str):
                    mask = df[age_col].astype(str) == val
                else:
                    mask = df[age_col] == val
                
                cnt = df[mask].shape[0]
                if cnt > 0:
                    log(f"- **Found Invalid Age**: `{val}` (Count: {cnt})")
            except:
                pass
                
        # Check decimals generally
        decimals = df[df[age_col].astype(str).str.match(r'^\d+\.\d+$', na=False)]
        if not decimals.empty:
             log(f"- **Found Decimal Ages**: Count {len(decimals)} (e.g., {decimals[age_col].iloc[0]})")

    # 4. Image Dimensions (Slanted/Landscape)
    log("\n### Image Dimensions")
    if 'OriginalImageWidth' in df.columns and 'OriginalImageHeight' in df.columns:
        slanted = df[df['OriginalImageWidth'] > df['OriginalImageHeight']]
        log(f"- **Landscape/Slanted Images**: {len(slanted)} (Width > Height)")

    with open('entry_report_specific.md', 'w') as f:
        f.write('\n'.join(report))
        print("Report written to entry_report_specific.md")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = r"c:\Users\ASUS\Documents\Techtatva\raw_data\Data_Entry.csv"
    analyze_entry(file_path)
