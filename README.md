# Medical Data Cleaning & Analysis

This repository contains the final scripts and reports for cleaning `BBox_List.csv` and `Data_Entry.csv` for Machine Learning usage.

## Active Files Explanation

### 1. Cleaning & Generation
-   **`clean_data_phase3.py`**: **The Core Cleaning Script**. 
    -   Loads raw CSVs.
    -   **Strict Rules**: Drops invalid geometries, filenames, and **Empty/Unknown Ages**.
    -   **Outputs**: 
        -   `cleaned_data/final_data_entry_v2.csv` (High Quality)
        -   `cleaned_data/final_bbox_list_v2.csv`
        -   `dropped_data/*_consolidated.csv` (Archived rows)
        -   `stats_report.json` (Summary of drops)
-   **`generate_insights.py`**: 
    -   Reads the *cleaned* CSVs.
    -   Generates visualization plots (`insight_finding_distribution.png`, `insight_disease_by_gender.png`).

### 2. Reports
-   **`data_inconsistency_report.md`**: The definitive audit report. Details every inconsistency found (invalid images, spelling errors, numeric ages) and confirms their resolution status.

### 3. Analysis Tools (Validation)
-   **`analyze_data_entry.py`**: Run this to verify `Data_Entry.csv` against rules. Used to generate the inconsistency report.
-   **`analyze_bbox.py`**: Run this to verify `BBox_List.csv` (checks negative coordinates, missing images).
-   **`analyze_images.py`**: Checks physical image folders. (Note: Currently reports 100% missing because raw images are not flattened).

### 4. Utilities
-   **`cleanup.py`**: Utility to remove temporary/intermediate files.
-   **`requirements.txt`**: Python dependencies.

## How to Run
1.  **Clean Data**: `python clean_data_phase3.py`
2.  **Generate Plots**: `python generate_insights.py`
