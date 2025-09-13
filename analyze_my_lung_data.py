
# Quick script to analyze your lung dataset
# Replace the data path with your actual lung dataset path

from lung_data_analyzer import analyze_lung_dataset

# Common lung dataset paths (adjust as needed)
possible_paths = [
    "../data/Task06_Lung",          # Common Medical Segmentation Decathlon format
    "../data/TaskXX_Lung",          # Replace XX with your task number
    "../data/lung_data",            # Simple folder name
    "../data/Lung",                 # Capital L
    "../data/lungs",                # Plural
]

# Try to find your lung dataset
import os
from pathlib import Path

print("🔍 Looking for lung dataset...")
lung_path = None

for path in possible_paths:
    if Path(path).exists():
        print(f"✅ Found dataset at: {path}")
        lung_path = path
        break
    else:
        print(f"❌ Not found: {path}")

if lung_path is None:
    print("\n❓ Dataset not found in common locations.")
    print("Please update the path in this script or create the folder structure:")
    print("  YourDataFolder/")
    print("  ├── imagesTr/")
    print("  │   ├── lung_001.nii.gz")
    print("  │   ├── lung_002.nii.gz")
    print("  │   └── ...")
    print("  └── labelsTr/")
    print("      ├── lung_001.nii.gz") 
    print("      ├── lung_002.nii.gz")
    print("      └── ...")

    # Prompt user for path
    custom_path = input("\nEnter your lung dataset path (or press Enter to skip): ").strip()
    if custom_path and Path(custom_path).exists():
        lung_path = custom_path
    else:
        print("Exiting...")
        exit()

# Run the analysis
print(f"\n🚀 Starting analysis of: {lung_path}")
results, df = analyze_lung_dataset(lung_path)

if results:
    print("\n✅ Analysis completed successfully!")
    print("\nNext steps for your lung anomaly detection project:")
    print("1. Review the analysis results and visualizations")
    print("2. Adjust anomaly detection criteria if needed")
    print("3. Create lung_preprocessing.py (similar to spleen_preprocessing.py)")
    print("4. Build lung_3d_model.py for the autoencoder architecture")
    print("5. Create training pipeline for lung anomaly detection")
else:
    print("\n❌ Analysis failed. Check your data path and file structure.")
