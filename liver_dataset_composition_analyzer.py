import nibabel as nib
import numpy as np
from pathlib import Path
from liver_preprocessing import LiverDataPreprocessor

class LiverDatasetAnalyzer:
    def __init__(self):
        self.preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
        print(f"📊 Found {len(self.preprocessor.image_files)} liver image files")
        print(f"📊 Found {len(self.preprocessor.label_files)} liver label files")

    def analyze_liver_labels_composition(self):
        """Analyze EXACT composition of liver dataset labels"""
        print("\n🔬 ANALYZING LIVER DATASET COMPOSITION")
        print("="*70)
        print("Task03_Liver Label Structure:")
        print("  Label 0: Background")
        print("  Label 1: Normal liver tissue") 
        print("  Label 2: Liver tumor/lesion")
        print("="*70)
        
        normal_liver_only = 0      # Only label 0,1 (normal liver)
        liver_with_tumors = 0      # Has label 0,1,2 (liver + tumors)  
        other_cases = 0            # Unusual label patterns
        
        detailed_results = []
        
        total_files = len(self.preprocessor.image_files)
        
        for i in range(total_files):
            try:
                image_path = self.preprocessor.image_files[i]
                mask_path = self.preprocessor.label_files[i]
                
                print(f"\n📋 Volume {i+1}/{total_files}: {image_path.name}")
                
                # Load mask to check labels
                mask = nib.load(mask_path).get_fdata()
                unique_labels = np.unique(mask)
                
                # Count voxels for each label
                label_counts = {}
                for label in unique_labels:
                    count = np.sum(mask == label)
                    percentage = count / mask.size * 100
                    label_counts[int(label)] = {
                        'count': count, 
                        'percentage': percentage
                    }
                
                print(f"  Unique labels: {unique_labels}")
                for label, info in label_counts.items():
                    if label > 0:  # Skip background
                        print(f"    Label {label}: {info['count']:,} voxels ({info['percentage']:.2f}%)")
                
                # Classify the volume
                has_liver = 1 in unique_labels
                has_tumor = 2 in unique_labels
                
                if has_liver and not has_tumor:
                    # Only normal liver tissue
                    normal_liver_only += 1
                    category = "✅ NORMAL LIVER ONLY"
                    
                elif has_liver and has_tumor:
                    # Liver with tumors
                    liver_with_tumors += 1 
                    category = "🚨 LIVER WITH TUMORS"
                    
                else:
                    # Unusual case
                    other_cases += 1
                    category = "❓ UNUSUAL LABELS"
                
                print(f"  Classification: {category}")
                
                # Store detailed results
                detailed_results.append({
                    'index': i+1,
                    'filename': image_path.name,
                    'labels': list(unique_labels),
                    'has_liver': has_liver,
                    'has_tumor': has_tumor,
                    'category': category,
                    'liver_voxels': label_counts.get(1, {}).get('count', 0),
                    'tumor_voxels': label_counts.get(2, {}).get('count', 0)
                })
                
            except Exception as e:
                print(f"  ❌ Error analyzing {image_path.name}: {e}")
                continue
        
        # Print summary
        total_analyzed = len(detailed_results)
        normal_percentage = (normal_liver_only / total_analyzed * 100) if total_analyzed > 0 else 0
        tumor_percentage = (liver_with_tumors / total_analyzed * 100) if total_analyzed > 0 else 0
        
        print(f"\n📊 LIVER DATASET COMPOSITION SUMMARY")
        print("="*70)
        print(f"Total volumes analyzed: {total_analyzed}")
        print(f"✅ Normal liver only: {normal_liver_only} ({normal_percentage:.1f}%)")
        print(f"🚨 Liver with tumors: {liver_with_tumors} ({tumor_percentage:.1f}%)")  
        print(f"❓ Other cases: {other_cases}")
        
        print(f"\n🎯 IMPLICATIONS FOR YOUR MODEL:")
        print("="*70)
        
        if liver_with_tumors > normal_liver_only:
            print("⚠️ MAJOR ISSUE FOUND:")
            print(f"  - Your model was trained on MOSTLY TUMOROUS LIVER ({tumor_percentage:.1f}%)")
            print(f"  - Only {normal_percentage:.1f}% were normal liver")
            print(f"  - Model learned to reconstruct TUMORS as 'normal'!")
            print(f"  - This is why synthetic pathology shows as 'normal'")
            
            print(f"\n💡 SOLUTIONS:")
            print(f"  1. Retrain model on ONLY normal liver volumes (indices with label 1 only)")
            print(f"  2. Or use much more sensitive threshold")
            print(f"  3. Or implement tumor-aware training")
            
        else:
            print("✅ Dataset composition is reasonable for anomaly detection")
            print(f"  - Majority ({normal_percentage:.1f}%) are normal liver")
            print(f"  - Model should work well with proper threshold")
        
        return detailed_results, normal_liver_only, liver_with_tumors

    def list_normal_liver_volumes_only(self):
        """List volumes that contain ONLY normal liver (no tumors)"""
        print(f"\n📋 IDENTIFYING PURE NORMAL LIVER VOLUMES")
        print("="*70)
        
        normal_only_indices = []
        
        for i in range(len(self.preprocessor.image_files)):
            try:
                mask_path = self.preprocessor.label_files[i]
                mask = nib.load(mask_path).get_fdata()
                unique_labels = np.unique(mask)
                
                # Check if it has liver (1) but NO tumors (2)
                has_liver = 1 in unique_labels
                has_tumor = 2 in unique_labels
                
                if has_liver and not has_tumor:
                    normal_only_indices.append(i)
                    print(f"✅ Volume {i+1}: {self.preprocessor.image_files[i].name}")
                    print(f"    Labels: {unique_labels}")
                    
            except Exception as e:
                print(f"❌ Error checking volume {i+1}: {e}")
                continue
        
        print(f"\n📊 PURE NORMAL LIVER VOLUMES:")
        print(f"Found {len(normal_only_indices)} volumes with ONLY normal liver tissue")
        print(f"Indices: {normal_only_indices}")
        
        if len(normal_only_indices) < 20:
            print(f"\n⚠️ WARNING: Very few pure normal liver volumes!")
            print(f"This explains why your model has issues with threshold.")
        
        return normal_only_indices

    def analyze_tumor_characteristics(self):
        """Analyze characteristics of liver tumors in dataset"""
        print(f"\n🩺 ANALYZING LIVER TUMOR CHARACTERISTICS")
        print("="*70)
        
        tumor_volumes = []
        
        for i in range(len(self.preprocessor.image_files)):
            try:
                image_path = self.preprocessor.image_files[i]
                mask_path = self.preprocessor.label_files[i]
                
                mask = nib.load(mask_path).get_fdata()
                unique_labels = np.unique(mask)
                
                if 2 in unique_labels:  # Has tumors
                    liver_voxels = np.sum(mask == 1)
                    tumor_voxels = np.sum(mask == 2)
                    tumor_percentage = tumor_voxels / (liver_voxels + tumor_voxels) * 100
                    
                    tumor_volumes.append({
                        'index': i+1,
                        'filename': image_path.name,
                        'liver_voxels': liver_voxels,
                        'tumor_voxels': tumor_voxels,
                        'tumor_percentage': tumor_percentage
                    })
                    
                    print(f"🚨 Volume {i+1}: {image_path.name}")
                    print(f"    Liver voxels: {liver_voxels:,}")
                    print(f"    Tumor voxels: {tumor_voxels:,}")
                    print(f"    Tumor burden: {tumor_percentage:.1f}%")
                    
            except Exception as e:
                continue
        
        if tumor_volumes:
            avg_tumor_percentage = np.mean([v['tumor_percentage'] for v in tumor_volumes])
            print(f"\n📊 TUMOR STATISTICS:")
            print(f"Volumes with tumors: {len(tumor_volumes)}")
            print(f"Average tumor burden: {avg_tumor_percentage:.1f}%")
        
        return tumor_volumes

def main():
    """Run complete liver dataset analysis"""
    print("🫀 COMPREHENSIVE LIVER DATASET ANALYSIS")
    print("="*80)
    
    try:
        analyzer = LiverDatasetAnalyzer()
        
        # Step 1: Analyze dataset composition 
        results, normal_count, tumor_count = analyzer.analyze_liver_labels_composition()
        
        # Step 2: List pure normal liver volumes
        normal_indices = analyzer.list_normal_liver_volumes_only()
        
        # Step 3: Analyze tumor characteristics
        tumor_info = analyzer.analyze_tumor_characteristics()
        
        # Step 4: Final recommendations
        print(f"\n🎯 FINAL ANALYSIS & RECOMMENDATIONS")
        print("="*80)
        
        total_analyzed = normal_count + tumor_count
        if tumor_count > normal_count:
            print(f"⚠️ CRITICAL FINDING:")
            print(f"  - Dataset is {tumor_count/total_analyzed*100:.1f}% tumorous liver cases")
            print(f"  - Only {normal_count/total_analyzed*100:.1f}% are pure normal liver")
            print(f"  - Your model learned tumors as 'normal' reconstruction targets!")
            
            print(f"\n💡 RECOMMENDED FIXES:")
            print(f"  1. RETRAIN MODEL using only normal liver volumes: {normal_indices}")
            print(f"  2. Or use MUCH lower threshold (try 0.005 - 0.010)")
            print(f"  3. Or implement tumor-aware preprocessing")
            
        else:
            print(f"✅ Dataset composition is suitable for anomaly detection")
            
        # Save results
        with open("../results/liver_dataset_analysis.txt", "w") as f:
            f.write(f"Liver Dataset Analysis Results\n")
            f.write(f"==============================\n")
            f.write(f"Total volumes: {total_analyzed}\n")
            f.write(f"Normal liver only: {normal_count}\n")
            f.write(f"Liver with tumors: {tumor_count}\n")
            f.write(f"Pure normal indices: {normal_indices}\n")
        
        print(f"\n✅ Results saved to: ../results/liver_dataset_analysis.txt")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
