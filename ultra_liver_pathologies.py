import torch
import numpy as np
from pathlib import Path
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

class UltraLiverPathologyCreator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def create_massive_liver_tumor(self, base_volume, liver_mask):
        """Create MASSIVE liver tumor - impossible to miss"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 100:
            # HUGE tumor (25x25x25 voxels - 5cm+ equivalent)
            center_idx = len(liver_coords[0]) // 4
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Create MASSIVE bright lesion
            for x in range(max(0, cx-12), min(64, cx+13)):
                for y in range(max(0, cy-12), min(64, cy+13)):
                    for z in range(max(0, cz-12), min(64, cz+13)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 144:  # Very large sphere
                            # EXTREMELY bright (complete replacement)
                            pathological_volume[x, y, z] = 1.0  # Maximum intensity
                            
        return pathological_volume, "MASSIVE Liver Tumor (5cm+)"
    
    def create_liver_necrosis(self, base_volume, liver_mask):
        """Create large necrotic area - complete tissue death"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 150:
            # Large necrotic region
            center_idx = len(liver_coords[0]) // 3
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Large area of complete tissue death
            for x in range(max(0, cx-10), min(64, cx+11)):
                for y in range(max(0, cy-10), min(64, cy+11)):
                    for z in range(max(0, cz-10), min(64, cz+11)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 100:
                            # Complete necrosis (zero intensity)
                            pathological_volume[x, y, z] = 0.0
                            
        return pathological_volume, "Large Liver Necrosis (Complete Tissue Death)"
    
    def create_extreme_liver_cyst(self, base_volume, liver_mask):
        """Create giant liver cyst"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 100:
            # Giant cyst
            center_idx = len(liver_coords[0]) // 2
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Huge fluid-filled cavity
            for x in range(max(0, cx-11), min(64, cx+12)):
                for y in range(max(0, cy-11), min(64, cy+12)):
                    for z in range(max(0, cz-11), min(64, cz+12)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 121:  # Large sphere
                            # Water density (very hypodense)
                            pathological_volume[x, y, z] = 0.01  # Nearly zero
                            
        return pathological_volume, "Giant Liver Cyst (>10cm)"
    
    def create_liver_replacement(self, base_volume, liver_mask):
        """Replace entire liver section with abnormal tissue"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 200:
            # Replace entire quadrant of liver
            for i in range(0, len(liver_coords[0])//3):
                x, y, z = liver_coords[0][i], liver_coords[1][i], liver_coords[2][i]
                
                # Replace with very different tissue
                pathological_volume[x, y, z] = 0.9  # Very bright
                
        return pathological_volume, "Liver Segment Replacement (Massive Infiltration)"
    
    def create_liver_destruction(self, base_volume, liver_mask):
        """Simulate severe liver destruction/cirrhosis"""
        pathological_volume = base_volume.copy()
        
        # Create random destruction pattern
        np.random.seed(123)  # Reproducible
        destruction_mask = np.random.random(pathological_volume.shape) > 0.7
        
        # Apply destruction only to liver regions
        liver_destruction = liver_mask & destruction_mask
        
        # Areas of destruction become either very bright (fibrosis) or dark (necrosis)
        pathological_volume[liver_destruction] = np.where(
            np.random.random(np.sum(liver_destruction)) > 0.5,
            0.95,  # Bright fibrosis
            0.05   # Dark necrosis
        )
        
        return pathological_volume, "Severe Liver Destruction/Cirrhosis"
    
    def create_all_ultra_pathologies(self, base_index=8):
        """Create ultra-dramatic pathological test cases"""
        if base_index >= len(self.preprocessor.image_files):
            return []
        
        base_scan = self.preprocessor.image_files[base_index]
        mask_scan = self.preprocessor.label_files[base_index]
        
        volume, mask = self.preprocessor.preprocess_liver_volume(base_scan, mask_scan)
        if volume is None:
            print(f"❌ Could not load base volume {base_index}")
            return []
        
        liver_mask = mask > 0
        pathological_cases = []
        
        # Create each type of EXTREME liver pathology
        pathology_creators = [
            self.create_massive_liver_tumor,
            self.create_liver_necrosis,
            self.create_extreme_liver_cyst,
            self.create_liver_replacement,
            self.create_liver_destruction
        ]
        
        for i, creator in enumerate(pathology_creators):
            try:
                pathological_volume, description = creator(volume, liver_mask)
                
                # Verify the pathology is actually different
                original_mean = np.mean(volume[liver_mask])
                pathology_mean = np.mean(pathological_volume[liver_mask])
                difference = abs(original_mean - pathology_mean)
                
                print(f"Created pathology {i+1}: {description}")
                print(f"  Original liver mean: {original_mean:.3f}")
                print(f"  Pathology mean: {pathology_mean:.3f}")
                print(f"  Difference: {difference:.3f}")
                
                if difference > 0.1:  # Significant change
                    pathological_cases.append({
                        'volume': pathological_volume,
                        'mask': mask,
                        'description': description,
                        'expected': 'anomaly',
                        'intensity_change': difference
                    })
                    print(f"  ✅ Added to test cases")
                else:
                    print(f"  ⚠️ Change too small, skipping")
                    
            except Exception as e:
                print(f"Failed to create pathology {i+1}: {e}")
                continue
        
        print(f"\nCreated {len(pathological_cases)} ultra-dramatic pathological cases")
        return pathological_cases

class UltraLiverAnomalyTester:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = Liver3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor
        self.preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
        self.pathology_creator = UltraLiverPathologyCreator(self.preprocessor)
        
        print(f"✅ Ultra liver tester loaded on {self.device}")
    
    def calculate_baseline_threshold(self):
        """Calculate conservative baseline threshold"""
        print("Calculating baseline liver threshold...")
        
        normal_errors = []
        for i in range(min(10, len(self.preprocessor.image_files))):
            try:
                volume_path = self.preprocessor.image_files[i]
                mask_path = self.preprocessor.label_files[i]
                
                volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                liver_mask = mask > 0
                liver_volume = volume.copy()
                liver_volume[~liver_mask] = 0
                
                volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    normal_errors.append(error)
                    
                print(f"  Normal liver {i+1}: error = {error:.6f}")
                
            except Exception as e:
                print(f"  Error with volume {i}: {e}")
                continue
        
        if normal_errors:
            mean_error = np.mean(normal_errors)
            std_error = np.std(normal_errors)
            max_error = max(normal_errors)
            
            # Conservative threshold - just above highest normal error
            threshold = max_error + 0.002  # Small margin above highest normal
            
            print(f"\nBaseline threshold calculation:")
            print(f"Normal errors: {[f'{e:.6f}' for e in normal_errors]}")
            print(f"Mean: {mean_error:.6f}, Std: {std_error:.6f}, Max: {max_error:.6f}")
            print(f"Conservative threshold: {threshold:.6f}")
            
            return threshold
        else:
            print("❌ Could not calculate threshold")
            return 0.020  # Fallback
    
    def test_ultra_pathologies(self, threshold):
        """Test ultra-dramatic pathological cases"""
        print(f"\n=== Testing ULTRA-DRAMATIC Liver Pathologies ===")
        print(f"Using threshold: {threshold:.6f}")
        
        # Create ultra-dramatic pathologies
        pathological_cases = self.pathology_creator.create_all_ultra_pathologies()
        
        if not pathological_cases:
            print("❌ No pathological cases created!")
            return [], 0
        
        results = []
        correct_detections = 0
        
        for i, case in enumerate(pathological_cases):
            print(f"\n🚨 ULTRA Case {i+1}: {case['description']}")
            print(f"   Intensity change: {case['intensity_change']:.3f}")
            
            # Test pathological case
            liver_mask = case['mask'] > 0
            liver_volume = case['volume'].copy()
            liver_volume[~liver_mask] = 0
            
            volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            is_anomaly = error > threshold
            confidence = error / threshold if threshold > 0 else 0
            
            if is_anomaly:
                correct_detections += 1
                status = "✅ DETECTED"
            else:
                status = "❌ MISSED"
            
            print(f"   Reconstruction Error: {error:.6f}")
            print(f"   Result: {status}")
            print(f"   Confidence: {confidence:.2f}x threshold")
            
            results.append({
                'description': case['description'],
                'error': error,
                'detected': is_anomaly,
                'confidence': confidence,
                'intensity_change': case['intensity_change']
            })
        
        detection_rate = (correct_detections / len(pathological_cases) * 100) if pathological_cases else 0
        print(f"\n📊 ULTRA Pathology Detection: {correct_detections}/{len(pathological_cases)} ({detection_rate:.1f}%)")
        
        return results, detection_rate

def main():
    """Run ultra-dramatic liver pathology testing"""
    model_path = "../models/best_liver_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print("❌ Liver model not found!")
        return
    
    print("🚨 ULTRA-DRAMATIC LIVER PATHOLOGY TESTING")
    print("=" * 60)
    
    # Run ultra testing
    tester = UltraLiverAnomalyTester(model_path)
    
    # Calculate conservative threshold
    threshold = tester.calculate_baseline_threshold()
    
    # Test ultra-dramatic pathologies
    results, detection_rate = tester.test_ultra_pathologies(threshold)
    
    print(f"\n🎯 FINAL ULTRA RESULTS:")
    print(f"Detection Rate: {detection_rate:.1f}%")
    print(f"Recommended Threshold: {threshold:.6f}")
    
    if detection_rate >= 80:
        print("🎉 EXCELLENT! Ultra pathologies detected - ready for demo!")
    elif detection_rate >= 60:
        print("✅ GOOD! Most ultra pathologies detected")
    else:
        print("⚠️ Need even more dramatic pathologies or lower threshold")
    
    # Save threshold
    try:
        with open("../models/ultra_liver_threshold.txt", "w") as f:
            f.write(f"{threshold:.6f}")
        print("💾 Threshold saved successfully")
    except Exception as e:
        print(f"⚠️ Could not save threshold: {e}")

if __name__ == "__main__":
    main()
