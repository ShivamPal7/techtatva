import os
import glob

def cleanup_zero_byte_files(root_dir):
    print(f"Scanning {root_dir} for zero-byte files...")
    count = 0
    # Recursive search for all files
    for filepath in glob.iglob(os.path.join(root_dir, '**/*'), recursive=True):
        if os.path.isfile(filepath):
            try:
                if os.path.getsize(filepath) == 0:
                    print(f"Deleting 0-byte file: {filepath}")
                    os.remove(filepath)
                    count += 1
            except OSError as e:
                print(f"Error accessing {filepath}: {e}")
    
    print(f"Cleanup complete. Deleted {count} files.")

if __name__ == "__main__":
    # Target the raw_data directory
    target_dir = os.path.abspath('raw_data')
    if os.path.exists(target_dir):
        cleanup_zero_byte_files(target_dir)
    else:
        print(f"Directory {target_dir} not found.")
