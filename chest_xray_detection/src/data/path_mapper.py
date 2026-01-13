import os
import pandas as pd
from glob import glob


def map_image_paths(base_dir, csv_path, output_path):
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Scanning for images in {base_dir}...")
    # Recursively find all png files
    # The user has images in raw_data/images which contains images_001, etc.
    image_paths = glob(os.path.join(base_dir, '**', '*.png'), recursive=True)
    
    print(f"Found {len(image_paths)} images.")
    
    # Create a dictionary for fast lookup: filename -> full_path
    path_dict = {os.path.basename(p): p for p in image_paths}
    
    # Map paths to the dataframe
    print("Mapping paths to dataframe...")
    df['path'] = df['Image Index'].map(path_dict)
    
    # Check for missing images
    missing = df['path'].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} images from CSV were not found on disk!")
        # Optional: drop missing
        # df = df.dropna(subset=['path'])
    else:
        print("All images found successfully!")
        
    print(f"Saving updated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Define paths relative to the script execution or absolute
    # Assuming script is run from project root (Techtatva)
    BASE_IMAGE_DIR = os.path.abspath("raw_data/images")
    INPUT_CSV = os.path.abspath("cleaned_data/final_data_entry_v2.csv")
    OUTPUT_CSV = os.path.abspath("cleaned_data/dataset_with_paths.csv")
    
    map_image_paths(BASE_IMAGE_DIR, INPUT_CSV, OUTPUT_CSV)
