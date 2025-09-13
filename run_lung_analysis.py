#!/usr/bin/env python3
"""
Simple runner script for lung data analysis
Works on Windows and automatically finds common lung dataset locations
"""

import sys
from pathlib import Path
import os

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

try:
    from lung_data_analyzer import analyze_lung_dataset
    print("✅ Successfully imported lung analyzer")
except ImportError as e:
    print(f"❌ Could not import lung_data_analyzer: {e}")
    print("Make sure lung_data_analyzer.py is in the same directory")
    input("Press Enter to exit...")
    sys.exit(1)

def main():
    print("🫁 LUNG DATASET ANALYZER - WINDOWS")
    print("=" * 50)
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Common lung dataset paths for Windows
    possible_paths = [
        # Relative to current liver project directory
        current_dir / ".." / ".." / "lung_3d_project" / "data",
        current_dir / ".." / "lung_data",
        current_dir / "lung_data",
        
        # Common Windows absolute paths
        Path("C:/Users/saura/unetp_3d_liver/lung_3d_project/data"),
        Path("C:/Data/Lung_Dataset"),
        Path("D:/Medical_Data/Lung"),
        Path("C:/Users/saura/Desktop/Lung_Data"),
        
        # Alternative naming
        current_dir / ".." / ".." / "lung_project" / "data",
        current_dir / ".." / "Task06_Lung",
        current_dir / ".." / "Lung_Dataset",
    ]
    
    print("\n🔍 Searching for lung dataset in common locations...")
    
    found_path = None
    for i, path in enumerate(possible_paths, 1):
        abs_path = path.resolve()
        print(f"{i:2d}. Checking: {abs_path}")
        
        if path.exists() and path.is_dir():
            # Check if it contains expected medical imaging structure
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            
            # Look for typical medical dataset structure
            has_images = any(subdir in subdirs for subdir in ["imagesTr", "images", "lung_images", "data"])
            has_labels = any(subdir in subdirs for subdir in ["labelsTr", "labels", "lung_labels", "masks"])
            
            if has_images and has_labels:
                print(f"    ✅ Found valid dataset structure!")
                found_path = str(path)
                break
            elif subdirs:
                print(f"    📁 Directory exists, subdirs: {subdirs[:3]}{'...' if len(subdirs) > 3 else ''}")
            else:
                print(f"    📁 Empty directory")
        else:
            print(f"    ❌ Does not exist")
    
    # If not found automatically, ask user
    if not found_path:
        print(f"\n❓ Lung dataset not found in expected locations.")
        print("Please provide the path to your lung dataset folder:")
        print("(It should contain imagesTr/ and labelsTr/ or images/ and labels/ subdirectories)")
        
        while True:
            user_path = input("\nEnter lung dataset path (or 'quit' to exit): ").strip().strip('"').strip("'")
            
            if user_path.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                return
            
            if user_path and Path(user_path).exists():
                found_path = user_path
                break
            else:
                print(f"❌ Path not found: {user_path}")
    
    # Run analysis
    if found_path:
        print(f"\n🚀 Starting analysis of: {found_path}")
        print("="*60)
        
        try:
            results, df = analyze_lung_dataset(found_path)
            
            if results:
                print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
                print("\n📊 What was analyzed:")
                print("- Lung tissue percentage in each scan")
                print("- Number of different tissue labels")
                print("- Classification as normal vs anomalous")
                print("- Detailed statistics saved to CSV")
                
                print(f"\n📁 Results saved to: lung_analysis_results.csv")
                print("\nNext steps for lung anomaly detection:")
                print("1. Review the analysis results")
                print("2. Adjust anomaly detection criteria if needed")
                print("3. Create lung preprocessing pipeline (like your liver project)")
                print("4. Build lung 3D autoencoder model")
                print("5. Train anomaly detection system")
                
            else:
                print("\n⚠️ Analysis completed but no valid results found")
                print("Check your dataset structure and file formats")
                
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print("\nCommon issues:")
            print("- Missing nibabel package: pip install nibabel")
            print("- Corrupted NIfTI files")
            print("- Incorrect dataset structure")
            print("- Insufficient disk space")
            
            import traceback
            print(f"\nFull error details:")
            traceback.print_exc()
    
    print(f"\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()