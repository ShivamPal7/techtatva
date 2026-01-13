# Data Inconsistency Report

**Date**: 2026-01-13
**Scope**: `raw_data/BBox_List.csv` & `raw_data/Data_Entry.csv`
**Status**: Critical Inconsistencies Identified & Fixed in Phase 3.

---

## 1. BBox_List.csv (Sahir)

| Rule / Check | Finding | Status |
| :--- | :--- | :--- |
| **Column Headers** | Headers had whitespace (`bbox_x `) and inconsistent separators (`bbox-y`). | **Fixed** (Standardized to `bbox_x`, `bbox_y`) |
| **Image Index: `missing_image.png`** | **Confirmed**. Found **148** rows with `missing_image.png`. | **Fixed** (Rows Dropped) |
| **Finding Label: Spellings** | Analyzed 984 rows. found only valid labels: `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltrate`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`. **No misspellings found in BBox file itself**, but enforced whitelist. | **Verified** |
| **bbox_x/y vs Image Dimensions** | **Confirmed**. Found rows where bbox extends beyond image (e.g., `Max Height 9999.0`). | **Fixed** (Validated against Metadata) |
| **Negative Values** | **Confirmed**. Found **98** rows with negative width (e.g., `-50.0`). | **Fixed** (Rows Dropped) |
| **Width/Height Logic** | **Confirmed**. Found rows with invalid geometry (Height 9999.0). | **Fixed** (Rows Dropped) |

---

## 2. Data_Entry.csv (Adarsh & Dev)

| Rule / Check | Finding | Status |
| :--- | :--- | :--- |
| **Column Headers** | Headers had whitespace (e.g., `patient_age `). | **Fixed** (Stripped) |
| **Gender Format** | **Confirmed**. Found `female `, ` Male`, `unknown`. | **Fixed** (Standardized to M/F, "Unknown" **Dropped**) |
| **Image Index: `invalid_image.png`** | **Confirmed**. Found **2,875** rows. | **Fixed** (Rows Dropped) |
| **Image Index: Case Sensitivity** | **Confirmed**. Found **2,933** files ending in `.PNG`. | **Fixed** (Renamed to .png) |
| **Image Index: Missing/Bad Format** | **Confirmed**. Found specific corrupt entries:<br>- `IMG_00000117_000.png` (1)<br>- `00141_.png` (1)<br>- `00000376_011.PNG` (1) | **Fixed** (Rows Dropped) |
| **Finding Labels: Numeric (`123`)** | **Confirmed**. Found `123`. | **Fixed** (Mapped to "No Finding") |
| **Finding Labels: `XYZ_Disease`** | **Confirmed**. Found `XYZ_Disease`. | **Fixed** (Mapped to "No Finding") |
| **Finding Labels: `Unknown_Disorder`** | **Confirmed**. Found `Unknown_Disorder`. | **Fixed** (Mapped to "No Finding") |
| **Finding Labels: `None`** | **Confirmed**. Found `None`. | **Fixed** (Mapped to "No Finding") |
| **View Position: `SIDEWAYS`** | **Confirmed**. Found **2,311** rows with `SIDEWAYS`. | **Fixed** (Mapped to `Lateral`) |
| **Patient Age: Zero/999/-5/150** | **Confirmed**. Specific invalid ages found. | **Fixed** (Nullified) |
| **Patient Age: Words (`Twenty Five`)** | **Confirmed**. Found **1,611** entries of `Twenty Five`. | **Fixed** (Parsed to 25) |
| **Patient Age: `Unknown`** | **Confirmed**. Found **1,652** entries. | **Fixed** (Nullified) |
| **Patient Age: Decimal (`45.7`)** | **Confirmed**. Found **1,593** decimals (e.g., `45.7`). | **Fixed** (Rounded/Floored) |
| **Image Duplicates** | **Confirmed**. Found **5,328** duplicate Image Index entries. | **Note** (Valid data) |
| **Slanted Images** | **Confirmed**. Found **84,510** Landscape images. | **Insight** (Valid data) |

---

## 3. Image Analysis (Sahir/Dev)
Analysis of `raw_data/images`:
-   **Total Images in CSV**: ~109k
-   **Total Found in Folder**: **Only 6 items identified** (likely subdirectories like `images_001`, `images_002`... or corrupt files).
-   **Findings**:
    -   The `raw_data/images` folder **does not contain flat image files**. It appears to contain **unextracted tarball folders** or corrupt placeholders.
    -   **Critical**: Physical image validation failed. 100% of images are effectively "missing" from the expected flat structure.
    -   **Action Recommendation**: Extract all `images_XX` subfolders into a single `images` directory before training.

## 4. Cleaning Statistics & Archives
**Cleanup Strategy**: "High-Fidelity / Strict". Rows with *any* critical issue (including missing Age) were dropped to ensure data purity for ML.

**Dropped Data Summary**:
-   **Total Data Entry Drops**: 18,009 rows
    -   *Reasons*: Invalid Age, Unknown Gender, Invalid Images (Regex Failure not matching `00000000_000.png`).
    -   **Archive**: `dropped_data/dropped_data_entry_consolidated.csv`
-   **Total BBox Drops**: 384 rows
    -   *Reasons*: Geometry errors, Whitelist exclusions.
    -   **Archive**: `dropped_data/dropped_bbox_list_consolidated.csv`

**Output Structure**:
-   **Cleaned Data**: `cleaned_data/final_data_entry_v2.csv`, `cleaned_data/final_bbox_list_v2.csv`
-   **Dropped Data**: `dropped_data/*_consolidated.csv`

## 5. Conclusion
**Phase 3 Cleaning** has standardized the Metadata and Annotations.
-   **Gender**: "Unknown" rows dropped.
-   **Age**: Empty/Invalid rows using strict validation dropped.
-   **Structure**: Cleaned vs Dropped data separated.


