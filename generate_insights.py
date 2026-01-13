
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_insights():
    print("Generating Insights...")
    try:
        df = pd.read_csv(os.path.join("cleaned_data", "final_data_entry_v2.csv"))
    except:
        print("Could not find cleaned_data/final_data_entry_v2.csv")
        return

    # Pre-process labels (Split pipes)
    # Explode 'finding_labels' into one row per label
    df['label_list'] = df['finding_labels'].str.split('|')
    exploded = df.explode('label_list')
    
    # Filter out "No Finding" for clearer disease charts? Or keep it?
    # Usually "No Finding" dominates, might skew scale. Let's keep it but noted.
    
    # 1. Bar Graph: Finding Label Count
    plt.figure(figsize=(12, 6))
    sns.countplot(data=exploded, x='label_list', order=exploded['label_list'].value_counts().index)
    plt.title("Distribution of Finding Labels")
    plt.xlabel("Finding Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("insight_finding_distribution.png")
    print("Saved insight_finding_distribution.png")
    
    # 2. Grouped Bar Chart: Disease by Gender
    # Filter for top diseases to avoid clutter if too many
    # But vocabulary is small (~15).
    
    plt.figure(figsize=(14, 7))
    # Countplot with hue=Gender
    # Filter Gender to M/F only for cleaner plot
    gender_subset = exploded[exploded['Gender'].isin(['M', 'F'])]
    
    sns.countplot(data=gender_subset, x='label_list', hue='Gender', 
                  order=gender_subset['label_list'].value_counts().index)
    plt.title("Disease Prevalence by Gender")
    plt.xlabel("Finding Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig("insight_disease_by_gender.png")
    print("Saved insight_disease_by_gender.png")

if __name__ == "__main__":
    generate_insights()
