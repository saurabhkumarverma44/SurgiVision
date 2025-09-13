import os
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from collections import Counter
import re

class LungDataAnalyzer:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        
        # Try different folder structures
        possible_structures = [
            ("imagesTr", "labelsTr"),
            ("images", "labels"), 
            ("lung_images", "lung_labels"),
            ("data", "masks"),
        ]
        
        self.images_dir = None
        self.labels_dir = None
        
        for img_folder, label_folder in possible_structures:
            img_path = self.data_root / img_folder
            label_path = self.data_root / label_folder
            
            if img_path.exists() and label_path.exists():
                self.images_dir = img_path
                self.labels_dir = label_path
                print(f"✅ Found structure: {img_folder}/ and {label_folder}/")
                break
        
        if self.images_dir is None:
            print(f"❌ Could not find image/label folders in {self.data_root}")
            self.image_files = []
            self.label_files = []
            return
        
        # Get file lists
        extensions = ["*.nii.gz", "*.nii", "*.nrrd"]
        all_image_files = []
        all_label_files = []
        
        for ext in extensions:
            all_image_files.extend(list(self.images_dir.glob(ext)))
            all_label_files.extend(list(self.labels_dir.glob(ext)))
        
        # Filter out hidden files
        self.image_files = sorted([f for f in all_image_files 
                                  if not f.name.startswith(('.', '_', '~'))])
        self.label_files = sorted([f for f in all_label_files 
                                  if not f.name.startswith(('.', '_', '~'))])
        
        print(f"Found {len(self.image_files)} lung images")
        print(f"Found {len(self.label_files)} lung labels")
        
        # Verify matching pairs
        if self.image_files and self.label_files:
            self._verify_file_pairs()
    
    def _verify_file_pairs(self):
        """Verify that image and label files match"""
        def extract_case_id(filename):
            # Use proper raw strings for regex patterns
            patterns = [
                r'lung_(\d+)',    # lung_001.nii.gz
                r'case_(\d+)',    # case_001.nii.gz  
                r'(\d+)',         # 001.nii.gz
                r'lung(\d+)',     # lung001.nii.gz
            ]
            for pattern in patterns:
                match = re.search(pattern, filename.lower())
                if match:
                    return match.group(1)
            return filename.split('.')[0]
        
        try:
            image_ids = [extract_case_id(f.name) for f in self.image_files]
            label_ids = [extract_case_id(f.name) for f in self.label_files]
            matched = set(image_ids) & set(label_ids)
            
            if len(matched) == len(image_ids) == len(label_ids):
                print(f"✅ All {len(self.image_files)} pairs matched")
            else:
                print(f"⚠️ {len(matched)} matched pairs found")
                print(f"Sample image IDs: {image_ids[:3]}")
                print(f"Sample label IDs: {label_ids[:3]}")
        except Exception as e:
            print(f"Could not verify pairing: {e}")
    
    def analyze_lung_masks(self):
        """Main analysis function"""
        print("\n=== ANALYZING LUNG DATASET ===")
        
        results = []
        normal_count = 0
        anomalous_count = 0
        
        for i, (img_path, mask_path) in enumerate(zip(self.image_files, self.label_files)):
            print(f"\nCase {i+1}/{len(self.image_files)}: {img_path.name}")
            
            try:
                img = nib.load(str(img_path))
                mask = nib.load(str(mask_path))
                
                img_data = img.get_fdata()
                mask_data = mask.get_fdata()
                
                # Calculate statistics
                lung_voxels = np.sum(mask_data > 0)
                total_voxels = np.prod(mask_data.shape)
                lung_percentage = (lung_voxels / total_voxels) * 100 if total_voxels > 0 else 0
                unique_labels = np.unique(mask_data[mask_data > 0])
                
                # Anomaly detection criteria - adjust these based on your dataset
                anomaly_flags = {
                    'very_small': lung_percentage < 2,          # Less than 2% lung tissue
                    'very_large': lung_percentage > 70,         # More than 70% lung tissue
                    'multiple_structures': len(unique_labels) > 2,  # Multiple structures
                    'unusual_labels': any(label > 10 for label in unique_labels)  # Unusual label values
                }
                
                has_anomaly = any(anomaly_flags.values())
                anomaly_reasons = [k for k, v in anomaly_flags.items() if v]
                
                if has_anomaly:
                    anomalous_count += 1
                    status = "ANOMALOUS"
                else:
                    normal_count += 1
                    status = "NORMAL"
                
                print(f"  Shape: {img_data.shape}")
                print(f"  Lung: {lung_voxels:,} voxels ({lung_percentage:.2f}%)")
                print(f"  Labels: {list(unique_labels)}")
                print(f"  Status: {status}")
                if anomaly_reasons:
                    print(f"  Reasons: {', '.join(anomaly_reasons)}")
                
                results.append({
                    'case_id': i+1,
                    'filename': img_path.name,
                    'image_shape': img_data.shape,
                    'lung_voxels': lung_voxels,
                    'lung_percentage': lung_percentage,
                    'unique_labels': list(unique_labels),
                    'num_labels': len(unique_labels),
                    'status': status,
                    'has_anomaly': has_anomaly,
                    'anomaly_reason': ', '.join(anomaly_reasons) if anomaly_reasons else 'none'
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'case_id': i+1,
                    'filename': img_path.name,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        return results, normal_count, anomalous_count
    
    def create_summary_report(self, results, normal_count, anomalous_count):
        """Generate summary report"""
        print("\n" + "="*60)
        print("🫁 LUNG DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        total = len(results)
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        valid = total - errors
        
        print(f"\n📁 DATASET OVERVIEW:")
        print(f"Total cases: {total}")
        print(f"Valid cases: {valid}")
        print(f"Error cases: {errors}")
        
        if valid > 0:
            print(f"\n🫁 LUNG CASES DISTRIBUTION:")
            print(f"Normal cases: {normal_count} ({normal_count/valid*100:.1f}%)")
            print(f"Anomalous cases: {anomalous_count} ({anomalous_count/valid*100:.1f}%)")
            
            # Detailed statistics for valid cases
            valid_results = [r for r in results if r.get('status') != 'ERROR']
            if valid_results:
                percentages = [r['lung_percentage'] for r in valid_results]
                voxels = [r['lung_voxels'] for r in valid_results]
                labels = [r['num_labels'] for r in valid_results]
                
                print(f"\n📈 STATISTICS:")
                print(f"Lung percentage: {min(percentages):.1f}% to {max(percentages):.1f}% (avg: {np.mean(percentages):.1f}%)")
                print(f"Lung voxels: {min(voxels):,} to {max(voxels):,} (avg: {np.mean(voxels):,.0f})")
                print(f"Number of labels: {min(labels)} to {max(labels)} (avg: {np.mean(labels):.1f})")
                
                # Anomaly breakdown
                anomaly_results = [r for r in valid_results if r.get('has_anomaly', False)]
                if anomaly_results:
                    print(f"\n🔍 ANOMALY BREAKDOWN:")
                    all_reasons = []
                    for r in anomaly_results:
                        reasons = r.get('anomaly_reason', '').split(', ')
                        all_reasons.extend([reason for reason in reasons if reason != 'none'])
                    
                    reason_counts = Counter(all_reasons)
                    for reason, count in reason_counts.most_common():
                        print(f"  {reason}: {count} cases")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if valid == 0:
            print("❌ No valid cases - check data path and file formats")
        elif normal_count == 0:
            print("⚠️ All cases flagged as anomalous - consider adjusting criteria")
        elif anomalous_count == 0:
            print("⚠️ No anomalies found - criteria may be too strict")
        elif valid > 0:
            ratio = anomalous_count / valid
            if ratio > 0.8:
                print("⚠️ Very high anomaly rate (>80%)")
            elif ratio < 0.05:
                print("📋 Very low anomaly rate (<5%) - may need more sensitive detection")
            else:
                print("✅ Reasonable anomaly detection balance")
        
        return results
    
    def save_results(self, results, output_dir="."):
        """Save results to CSV"""
        output_path = Path(output_dir) / "lung_analysis_results.csv"
        
        df_data = []
        for r in results:
            if r.get('status') != 'ERROR':
                df_data.append({
                    'case_id': r['case_id'],
                    'filename': r['filename'],
                    'lung_voxels': r['lung_voxels'],
                    'lung_percentage': r['lung_percentage'],
                    'num_labels': r['num_labels'],
                    'unique_labels': str(r['unique_labels']),
                    'status': r['status'],
                    'has_anomaly': r['has_anomaly'],
                    'anomaly_reason': r.get('anomaly_reason', 'none')
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
            print(f"\n💾 Analysis results saved to: {output_path}")
            return df
        else:
            print("\n❌ No valid data to save")
            return None

def analyze_lung_dataset(data_path):
    """Main analysis function - entry point"""
    print("🫁 LUNG DATASET ANALYZER")
    print("="*50)
    
    analyzer = LungDataAnalyzer(data_path)
    
    if not analyzer.image_files:
        print("❌ No image files found. Check your data path.")
        print("\nExpected structure:")
        print("  YourLungFolder/")
        print("  ├── imagesTr/ (or images/)")
        print("  │   ├── lung_001.nii.gz")
        print("  │   └── ...")
        print("  └── labelsTr/ (or labels/)")
        print("      ├── lung_001.nii.gz")
        print("      └── ...")
        return None, None
    
    results, normal, anomalous = analyzer.analyze_lung_masks()
    analyzer.create_summary_report(results, normal, anomalous)
    df = analyzer.save_results(results)
    
    print(f"\n✅ Analysis completed!")
    print(f"Summary: {normal} normal, {anomalous} anomalous lung cases")
    
    return results, df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("Enter path to lung dataset: ").strip().strip('"')
    
    if dataset_path and Path(dataset_path).exists():
        results, df = analyze_lung_dataset(dataset_path)
    else:
        print("❌ Invalid or missing dataset path")
        print("Please provide a valid path to your lung dataset folder")