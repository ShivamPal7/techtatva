
import pandas as pd
import numpy as np

def analyze_bbox(file_path):
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

    # 1. Column Header Analysis
    log("\n## 1. Column Header Check")
    clean_cols = df.columns.str.strip()
    if any(df.columns != clean_cols):
        log("- **Issue**: Headers contain leading/trailing whitespace.")
        log(f"  - Original: `{list(df.columns)}`")
        log(f"  - Cleaned:  `{list(clean_cols)}`")
        df.columns = clean_cols
    else:
        log("- Headers look clean (no whitespace).")
    
    if 'bbox-y' in df.columns:
        log("- **Issue**: Column `bbox-y` uses hyphen instead of underscore.")
    
    # 2. Column-by-Column Analysis
    log("\n## 2. Column-by-Column Analysis")
    
    # Image Index
    col = 'Image Index'
    log(f"\n### Column: `{col}`")
    if col in df.columns:
        nulls = df[col].isnull().sum()
        log(f"- **Missing Values**: {nulls}")
        
        placeholders = df[df[col].astype(str).str.contains('missing', case=False, na=False)]
        if not placeholders.empty:
            log(f"- **Issue**: Found {len(placeholders)} rows with 'missing' in filename.")
            log("  - Examples:\n```\n" + str(placeholders[col].value_counts().head()) + "\n```")
        
        non_png = df[~df[col].astype(str).str.lower().str.endswith('.png')]
        if not non_png.empty:
             log(f"- **Issue**: Found {len(non_png)} rows not ending in .png.")
             log("  - Examples:\n```\n" + str(non_png[col].head()) + "\n```")

    # Finding Label
    col = 'Finding Label'
    log(f"\n### Column: `{col}`")
    if col in df.columns:
        unique_cnt = df[col].nunique()
        log(f"- **Unique Values**: {unique_cnt}")
        log("- **Value Counts**:\n```\n" + str(df[col].value_counts()) + "\n```")
        
        # Check for numeric or weird labels
        # Simple check: are they all alpha?
        # df[col].astype(str).str.isalpha() might fail on "Pleural_Thickening" (underscore)
        # Let's check for digits
        with_digits = df[df[col].astype(str).str.contains(r'\d', regex=True)]
        if not with_digits.empty:
            log(f"- **Issue**: Found {len(with_digits)} labels containing digits (potential garbage).")
            log("  - Examples:\n```\n" + str(with_digits[col].unique()) + "\n```")

    # Coordinates
    coord_cols = ['bbox_x', 'bbox-y', 'width', 'height']
    if 'bbox_y' in df.columns: coord_cols[1] = 'bbox_y'
    if 'bbox_x' in df.columns: coord_cols[0] = 'bbox_x' 

    log(f"\n### Coordinates Analysis")
    for c in coord_cols:
        if c not in df.columns:
            log(f"- Skipping `{c}`, not found.")
            continue
            
        log(f"\n#### Sub-Column: `{c}`")
        numeric_series = pd.to_numeric(df[c], errors='coerce')
        non_numeric = df[numeric_series.isna()]
        
        if not non_numeric.empty:
            log(f"- **Issue**: Found {len(non_numeric)} non-numeric values.")
        
        # Ranges
        negatives = numeric_series[numeric_series < 0]
        if not negatives.empty:
            log(f"- **Critical Issue**: Found {len(negatives)} negative values.")
            log("  - Examples:\n```\n" + str(negatives.head()) + "\n```")
            
        zeros = numeric_series[numeric_series == 0]
        if not zeros.empty:
            if c in ['width', 'height']:
                log(f"- **Critical Issue**: Found {len(zeros)} zero values (invalid for dimensions).")
            else:
                log(f"- Note: Found {len(zeros)} zero values.")

        log(f"- Stats: Min={numeric_series.min()}, Max={numeric_series.max()}, Mean={numeric_series.mean():.2f}")

    with open('bbox_report.md', 'w') as f:
        f.write('\n'.join(report))
        print("Report written to bbox_report.md")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = r"c:\Users\ASUS\Documents\Techtatva\raw_data\BBox_List.csv"
    analyze_bbox(file_path)
