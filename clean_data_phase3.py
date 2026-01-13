
import pandas as pd
import numpy as np
import os
import re
import json

class DataCleanerPhase3:
    def __init__(self, raw_dir='raw_data', output_dir='.'):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.summary = {
            "dropped_bbox": 0,
            "dropped_entry_rows": 0,
            "modified_entry_cells": 0,
            "archived_files": []
        }
        
        # Create directories
        self.clean_dir = os.path.join(output_dir, 'cleaned_data')
        self.drop_dir = os.path.join(output_dir, 'dropped_data')
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.drop_dir, exist_ok=True)

    def clean(self):
        print("Loading Data...")
        entry = pd.read_csv(os.path.join(self.raw_dir, 'Data_Entry.csv'))
        bbox = pd.read_csv(os.path.join(self.raw_dir, 'BBox_List.csv'))
        
        # Accumulators for dropped rows
        dropped_entry_list = []
        dropped_bbox_list = []
        
        # --- 1. Clean Data Entry ---
        entry.columns = entry.columns.str.strip()
        initial_rows = len(entry)
        
        # A. Image Index Cleaning
        # Fix Extensions
        if 'Image Index' in entry.columns:
            entry['Image Index'] = entry['Image Index'].astype(str).str.strip().str.replace('.PNG', '.png', regex=False)
        
        # Regex Filter (Strict)
        # Format: 00000047_005.png -> 8 digits, underscore, 3 digits, .png
        filename_pattern = re.compile(r'^\d{8}_\d{3}\.png$')
        
        def is_valid_filename(fname):
            # Fast check first
            if not fname.endswith('.png'): return False
            # Regex check
            return bool(filename_pattern.match(fname))
            
        valid_img_mask = entry['Image Index'].apply(is_valid_filename)
        dropped_imgs = entry[~valid_img_mask].copy()
        if not dropped_imgs.empty:
            dropped_imgs['Drop_Reason'] = 'Invalid Filename Format (Regex)'
            dropped_entry_list.append(dropped_imgs)
            
        entry = entry[valid_img_mask]
        
        # B. Gender
        def clean_gender(g):
            g = str(g).upper().strip()
            if g in ['M', 'MALE', 'MAN']: return 'M'
            if g in ['F', 'FEMALE', 'WOMAN']: return 'F'
            return 'Unknown' 
        entry['Gender'] = entry['Gender'].apply(clean_gender)
        
        # Strict Rule: Drop Unknown Gender
        unknown_gender_mask = entry['Gender'] == 'Unknown'
        dropped_gender = entry[unknown_gender_mask].copy()
        if not dropped_gender.empty:
            dropped_gender['Drop_Reason'] = 'Unknown Gender'
            dropped_entry_list.append(dropped_gender)
            
        entry = entry[~unknown_gender_mask]
        
        # C. Finding Labels
        def clean_labels(l):
            l = str(l).strip()
            if l in ['123', 'XYZ_Disease', 'Unknown_Disorder', 'None', 'nan']: return "No Finding"
            if re.match(r'^[0-9]+$', l): return "No Finding"
            return l
        entry['finding_labels'] = entry['finding_labels'].apply(clean_labels)
        
        # D. View Position
        def clean_view(v):
            v = str(v).upper().strip()
            if v == 'SIDEWAYS': return 'Lateral'
            if v in ['PA', 'AP', 'LATERAL']: return v.title() if v == 'LATERAL' else v
            return 'Unknown'
        entry['View Position'] = entry['View Position'].apply(clean_view)
        
        # E. Age
        def clean_age(a):
            s = str(a).strip()
            # Word Map
            if s == "Twenty Five": return 25
            if s == "Forty Five.Seven" or s == "45.7": return 45
            
            # Numeric Parsing
            try:
                # Remove YMD
                clean_s = re.sub(r'[YMD]', '', s)
                val = float(clean_s)
                val = int(val) # Floor
                
                # Logic Checks
                if val <= 0: return None # 0, -5
                if val > 120: return None # 150, 999
                return val
            except:
                return None # "Unknown", unparseable
        
        entry['patient_age'] = entry['patient_age'].apply(clean_age)
        
        # Strict Rule: Drop Empty/Invalid Age
        invalid_age_mask = entry['patient_age'].isna()
        dropped_age = entry[invalid_age_mask].copy()
        if not dropped_age.empty:
            dropped_age['Drop_Reason'] = 'Invalid/Empty Age'
            dropped_entry_list.append(dropped_age)
            
        entry = entry[~invalid_age_mask]
        
        self.summary["dropped_entry_rows"] = initial_rows - len(entry)
        self.summary["modified_entry_cells"] = entry['patient_age'].isnull().sum()
        self.final_entry = entry
        
        # --- 2. Clean BBox ---
        bbox.columns = bbox.columns.str.strip()
        bbox.rename(columns={'bbox-y': 'bbox_y', 'bbox_x': 'bbox_x'}, inplace=True)
        
        initial_bbox = len(bbox)
        
        # A. Whitelist Labels
        allowed_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        valid_label_mask = bbox['Finding Label'].isin(allowed_labels)
        dropped_labels = bbox[~valid_label_mask].copy()
        if not dropped_labels.empty:
            dropped_labels['Drop_Reason'] = 'Label Not in Whitelist'
            dropped_bbox_list.append(dropped_labels)
            
        bbox = bbox[valid_label_mask]
        
        # B. Image Index Checks
        missing_mask = bbox['Image Index'] == 'missing_image.png'
        dropped_missing = bbox[missing_mask].copy()
        if not dropped_missing.empty:
            dropped_missing['Drop_Reason'] = 'missing_image.png Placeholder'
            dropped_bbox_list.append(dropped_missing)
            
        bbox = bbox[~missing_mask]
        
        # C. Coordinates
        for c in ['bbox_x', 'bbox_y', 'width', 'height']:
            bbox[c] = pd.to_numeric(bbox[c], errors='coerce')
        
        # Check NaNs
        nan_mask = bbox[['bbox_x', 'bbox_y', 'width', 'height']].isna().any(axis=1)
        dropped_nan = bbox[nan_mask].copy()
        if not dropped_nan.empty:
            dropped_nan['Drop_Reason'] = 'Non-Numeric Coordinates'
            dropped_bbox_list.append(dropped_nan)
            
        bbox = bbox[~nan_mask]
        
        # Negative / Zero validation
        valid_geom_mask = (bbox['width'] > 0) & (bbox['height'] > 0) & (bbox['bbox_x'] >= 0) & (bbox['bbox_y'] >= 0)
        dropped_geom = bbox[~valid_geom_mask].copy()
        if not dropped_geom.empty:
            dropped_geom['Drop_Reason'] = 'Negative/Zero Dimensions'
            dropped_bbox_list.append(dropped_geom)
            
        bbox = bbox[valid_geom_mask]
        
        # D. Out of Bounds (Merge with Cleaned Entry)
        dims = self.final_entry[['Image Index', 'OriginalImageWidth', 'OriginalImageHeight']].drop_duplicates('Image Index')
        merged = bbox.merge(dims, on='Image Index', how='left')
        
        # Drop if metadata missing (safe ML)
        merged_has_meta = merged.dropna(subset=['OriginalImageWidth'])
        
        # Logic: x + w <= W, y + h <= H
        valid_bounds_mask = (
            (merged_has_meta['bbox_x'] + merged_has_meta['width'] <= merged_has_meta['OriginalImageWidth']) & 
            (merged_has_meta['bbox_y'] + merged_has_meta['height'] <= merged_has_meta['OriginalImageHeight'])
        )
        
        final_bbox = merged_has_meta[valid_bounds_mask][bbox.columns]
        
        # Capture drops from Bounds Check
        # Rows in 'bbox' that are NOT in 'final_bbox'
        # To do this cleanly, we can filter using index if preserved, or just diffing.
        # Simple way: The rows dropped by 'valid_bounds_mask' in merged state.
        dropped_bounds = merged_has_meta[~valid_bounds_mask][bbox.columns].copy()
        if not dropped_bounds.empty:
             dropped_bounds['Drop_Reason'] = 'Out of Bounds'
             dropped_bbox_list.append(dropped_bounds)
             
        # Also capture rows dropped due to missing metadata (merged vs bbox)
        # (Implicitly handled if we consider all steps, but might be minor 0 rows here)
        
        self.summary["dropped_bbox"] = initial_bbox - len(final_bbox)
        self.final_bbox = final_bbox
        
        # --- Save Cleaned ---
        print("Saving final datasets...")
        self.final_entry.to_csv(os.path.join(self.clean_dir, 'final_data_entry_v2.csv'), index=False)
        self.final_bbox.to_csv(os.path.join(self.clean_dir, 'final_bbox_list_v2.csv'), index=False)
        
        # --- Save Dropped (Consolidated) ---
        print("Saving dropped datasets...")
        if dropped_entry_list:
            full_dropped_entry = pd.concat(dropped_entry_list, ignore_index=True)
            full_dropped_entry.to_csv(os.path.join(self.drop_dir, 'dropped_data_entry_consolidated.csv'), index=False)
            self.summary['archived_files'].append('dropped_data_entry_consolidated.csv')
            
        if dropped_bbox_list:
            full_dropped_bbox = pd.concat(dropped_bbox_list, ignore_index=True)
            full_dropped_bbox.to_csv(os.path.join(self.drop_dir, 'dropped_bbox_list_consolidated.csv'), index=False)
            self.summary['archived_files'].append('dropped_bbox_list_consolidated.csv')
        
        # Convert summary to int for JSON serializable
        def convert_int64(o):
            if isinstance(o, np.integer): return int(o)
            raise TypeError
            
        print("Phase 3 Complete.")
        with open(os.path.join(self.output_dir, 'stats_report.json'), 'w') as f:
            json.dump(self.summary, f, indent=4, default=convert_int64)

if __name__ == "__main__":
    cleaner = DataCleanerPhase3(r"c:\Users\ASUS\Documents\Techtatva\raw_data", r"c:\Users\ASUS\Documents\Techtatva")
    cleaner.clean()
