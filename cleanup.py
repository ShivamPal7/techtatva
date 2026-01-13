import os
import glob
def cleanup():
    # Whitelist of files to KEEP in root
    keep_files = [
        'data_inconsistency_report.md', 
        'clean_data_phase3.py', 
        'generate_insights.py', 
        'analyze_data_entry.py',
        'analyze_bbox.py',
        'analyze_images.py',
        'requirements.txt',
        'README.md',
        'cleanup.py',
        'cleaned_data',
        'dropped_data',
        'stats_report.json',
        'insight_finding_distribution.png',
        'insight_disease_by_gender.png',
        'raw_data',
        '.git'
    ]
    
    # Files to DELETE (Root level clutter & Old Granular Dropped)
    delete_list = [
        'final_bbox_list_fixed.csv',
        'final_data_entry_fixed.csv',
        'final_bbox_list.csv',
        'final_data_entry.csv',
        'cleaned_data/final_data_entry.csv', # Remove old if conflicting
        'cleaned_data/final_bbox_list.csv',
        # Old granular dropped files
        'dropped_data/dropped_invalid_age.csv',
        'dropped_data/dropped_unknown_gender.csv',
        'dropped_data/dropped_invalid_images.csv',
        'dropped_data/dropped_bbox_geometry.csv',
        'dropped_data/dropped_bbox_invalid_labels.csv',
        'dropped_data/dropped_data_entry.csv', # Old name
        'dropped_data/dropped_bbox_list.csv'   # Old name
    ]
    
    deleted = []
    
    # 1. Delete specific targets
    for f in delete_list:
        if os.path.exists(f):
            try:
                os.remove(f)
                deleted.append(f)
            except Exception as e:
                print(f"Error deleting {f}: {e}")

    # 2. Cleanup artifacts matching patterns
    all_files = glob.glob("*")
    for f in all_files:
        if f not in keep_files and "raw_data" not in f and os.path.isfile(f):
             # Logic: Delete if its an old report or log
             if f.endswith('.md') and f != 'data_inconsistency_report.md' and f != 'README.md':
                 try:
                    os.remove(f)
                    deleted.append(f)
                 except: pass
                    
    print(f"Deleted {len(deleted)} intermediate files.")
    for d in deleted:
        print(f" - {d}")

if __name__ == "__main__":
    cleanup()
