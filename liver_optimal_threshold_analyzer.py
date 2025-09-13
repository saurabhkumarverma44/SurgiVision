import torch
import numpy as np
from pathlib import Path
from liver_anomaly_detector import Liver3DAnomalyDetector

class LiverOptimalAnalyzer:
    def __init__(self):
        model_path = "../models/best_liver_3d_autoencoder.pth"
        if not Path(model_path).exists():
            print("❌ Liver model not found!")
            return None
        
        self.detector = Liver3DAnomalyDetector(model_path)
        print(f"✅ Loaded liver model")
        print(f"📊 Found {len(self.detector.preprocessor.image_files)} liver training volumes")

    def calculate_optimal_liver_threshold(self):
        """Calculate EXACT optimal threshold like we did for spleen"""
        print("\n🔬 CALCULATING OPTIMAL LIVER THRESHOLD")
        print("="*60)
        
        normal_errors = []
        
        # Test ALL training volumes to get proper statistics
        total_volumes = len(self.detector.preprocessor.image_files)
        print(f"Analyzing all {total_volumes} liver training volumes...")
        
        for i in range(total_volumes):
            try:
                volume_path = self.detector.preprocessor.image_files[i]
                mask_path = self.detector.preprocessor.label_files[i]
                
                print(f"Volume {i+1}/{total_volumes}: {volume_path.name}")
                
                # Use EXACT same preprocessing as training
                volume, mask = self.detector.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                
                if volume is None:
                    print(f"  ❌ Failed to process")
                    continue
                
                # Create liver-only volume (like training)
                liver_mask = mask > 0
                liver_volume = volume.copy()
                liver_volume[~liver_mask] = 0  # Zero out non-liver regions
                
                volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.detector.device)
                
                # Calculate reconstruction error
                with torch.no_grad():
                    reconstructed = self.detector.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                
                normal_errors.append(error)
                print(f"  ✅ Error: {error:.6f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        if not normal_errors:
            print("❌ No valid liver volumes processed!")
            return 0.025, []
        
        # Calculate statistics
        mean_error = np.mean(normal_errors)
        std_error = np.std(normal_errors)
        min_error = np.min(normal_errors)
        max_error = np.max(normal_errors)
        
        # Calculate different threshold options
        thresholds = {
            'conservative': mean_error + 4.0 * std_error,  # Very specific
            'balanced': mean_error + 3.0 * std_error,      # Balanced (RECOMMENDED)
            'sensitive': mean_error + 2.5 * std_error,     # More sensitive
            'very_sensitive': mean_error + 2.0 * std_error # Very sensitive
        }
        
        print(f"\n📊 LIVER TRAINING DATA STATISTICS:")
        print(f"Total volumes processed: {len(normal_errors)}")
        print(f"Reconstruction errors:")
        print(f"  Mean: {mean_error:.6f}")
        print(f"  Std:  {std_error:.6f}")
        print(f"  Min:  {min_error:.6f}")
        print(f"  Max:  {max_error:.6f}")
        
        print(f"\n🎯 THRESHOLD OPTIONS:")
        for name, threshold in thresholds.items():
            print(f"  {name.capitalize():15}: {threshold:.6f}")
        
        # Recommend the balanced threshold
        optimal_threshold = thresholds['balanced']
        print(f"\n💡 RECOMMENDED THRESHOLD: {optimal_threshold:.6f}")
        
        return optimal_threshold, normal_errors, thresholds

    def analyze_all_liver_volumes_with_threshold(self, threshold):
        """Analyze ALL 131 liver volumes with the optimal threshold"""
        print(f"\n🔬 ANALYZING ALL LIVER VOLUMES WITH THRESHOLD {threshold:.6f}")
        print("="*60)
        
        normal_count = 0
        anomaly_count = 0
        results = []
        
        total_volumes = len(self.detector.preprocessor.image_files)
        
        for i in range(total_volumes):
            try:
                volume_path = self.detector.preprocessor.image_files[i]
                
                print(f"Volume {i+1}/{total_volumes}: {volume_path.name}")
                
                # Use training detection method
                result = self.detector.detect_anomaly_from_training_file(i, threshold)
                
                if result:
                    if result['is_anomaly']:
                        anomaly_count += 1
                        status = "🚨 ANOMALY"
                    else:
                        normal_count += 1
                        status = "✅ NORMAL"
                    
                    print(f"  Error: {result['reconstruction_error']:.6f} | {status}")
                    results.append(result)
                else:
                    print(f"  ❌ Processing failed")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Summary
        total_processed = len(results)
        anomaly_rate = (anomaly_count / total_processed * 100) if total_processed > 0 else 0
        normal_rate = (normal_count / total_processed * 100) if total_processed > 0 else 0
        
        print(f"\n📊 LIVER DATASET ANALYSIS SUMMARY:")
        print(f"Total volumes processed: {total_processed}")
        print(f"Normal volumes: {normal_count} ({normal_rate:.1f}%)")
        print(f"Anomaly volumes: {anomaly_count} ({anomaly_rate:.1f}%)")
        print(f"Threshold used: {threshold:.6f}")
        
        if anomaly_count == 0:
            print(f"✅ PERFECT! All training volumes show NORMAL (as expected)")
            print(f"✅ Threshold {threshold:.6f} is OPTIMAL for liver detection")
        elif anomaly_rate <= 5:
            print(f"✅ GOOD! Very low false positive rate ({anomaly_rate:.1f}%)")
            print(f"✅ Threshold {threshold:.6f} is acceptable")
        else:
            print(f"⚠️ WARNING! High false positive rate ({anomaly_rate:.1f}%)")
            print(f"💡 Consider increasing threshold to reduce false positives")
        
        return results, normal_count, anomaly_count

    def test_synthetic_pathology_with_optimal_threshold(self, threshold):
        """Test synthetic liver pathology with optimal threshold"""
        print(f"\n🩺 TESTING SYNTHETIC LIVER PATHOLOGY")
        print(f"Threshold: {threshold:.6f}")
        print("="*60)
        
        # Create synthetic liver pathology
        pathological_volume, mask = self.detector.create_liver_pathology_test(base_idx=5)
        
        if pathological_volume is not None:
            # Test pathological case
            liver_mask = mask > 0
            masked_pathology = pathological_volume.copy()
            masked_pathology[~liver_mask] = 0
            
            volume_tensor = torch.FloatTensor(masked_pathology[np.newaxis, np.newaxis, ...]).to(self.detector.device)
            
            with torch.no_grad():
                reconstructed = self.detector.model(volume_tensor)
                error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            is_anomaly = error > threshold
            confidence = error / threshold
            
            print(f"Synthetic HCC Analysis:")
            print(f"  Reconstruction Error: {error:.6f}")
            print(f"  Threshold: {threshold:.6f}")
            print(f"  Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '❌ MISSED (INCREASE SENSITIVITY)'}")
            print(f"  Confidence: {confidence:.2f}x threshold")
            
            return is_anomaly, error, confidence
        else:
            print("❌ Failed to create synthetic pathology")
            return False, 0, 0

    def comprehensive_liver_analysis(self):
        """Run complete liver analysis like we did for spleen"""
        print("🫀 COMPREHENSIVE LIVER MODEL ANALYSIS")
        print("="*80)
        
        # Step 1: Calculate optimal threshold
        optimal_threshold, normal_errors, all_thresholds = self.calculate_optimal_liver_threshold()
        
        # Step 2: Analyze all volumes with optimal threshold
        results, normal_count, anomaly_count = self.analyze_all_liver_volumes_with_threshold(optimal_threshold)
        
        # Step 3: Test synthetic pathology
        pathology_detected, pathology_error, pathology_confidence = self.test_synthetic_pathology_with_optimal_threshold(optimal_threshold)
        
        # Step 4: Final recommendations
        print(f"\n🎯 FINAL LIVER ANALYSIS RESULTS:")
        print("="*80)
        print(f"📊 Optimal Threshold: {optimal_threshold:.6f}")
        print(f"📊 Training Data: {normal_count} Normal, {anomaly_count} Anomaly")
        print(f"📊 False Positive Rate: {anomaly_count/(normal_count+anomaly_count)*100:.1f}%")
        print(f"📊 Synthetic Pathology: {'✅ DETECTED' if pathology_detected else '❌ MISSED'}")
        
        if anomaly_count == 0 and pathology_detected:
            print(f"\n🏆 EXCELLENT! Your liver model is PERFECT:")
            print(f"  ✅ 0% false positives on training data")
            print(f"  ✅ Correctly detects synthetic pathology")
            print(f"  ✅ Optimal threshold: {optimal_threshold:.6f}")
            
        elif anomaly_count <= 5 and pathology_detected:
            print(f"\n✅ GOOD! Your liver model is working well:")
            print(f"  ✅ Very low false positive rate")
            print(f"  ✅ Correctly detects synthetic pathology") 
            print(f"  ✅ Use threshold: {optimal_threshold:.6f}")
            
        else:
            print(f"\n⚠️ NEEDS TUNING:")
            if anomaly_count > 5:
                print(f"  ⚠️ High false positive rate - try: {all_thresholds['conservative']:.6f}")
            if not pathology_detected:
                print(f"  ⚠️ Missing pathology - try: {all_thresholds['sensitive']:.6f}")
        
        print(f"\n💡 UPDATE YOUR STREAMLIT APP:")
        print(f"Replace current_threshold with: {optimal_threshold:.6f}")
        
        return optimal_threshold

def main():
    """Run liver analysis"""
    try:
        analyzer = LiverOptimalAnalyzer()
        if analyzer.detector is not None:
            optimal_threshold = analyzer.comprehensive_liver_analysis()
            
            # Save the optimal threshold
            with open("../models/liver_optimal_threshold.txt", "w") as f:
                f.write(f"{optimal_threshold:.6f}")
            
            print(f"\n✅ Optimal threshold saved to: ../models/liver_optimal_threshold.txt")
            print(f"🔧 Use this threshold in your Streamlit app!")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
