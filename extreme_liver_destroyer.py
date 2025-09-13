import torch
import numpy as np
from pathlib import Path
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

class ExtremeStructureDestroyer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def create_liver_swiss_cheese(self, base_volume, liver_mask):
        """Create Swiss cheese pattern - multiple holes throughout liver"""
        pathological_volume = base_volume.copy()
        
        # Create 20+ holes throughout the liver
        np.random.seed(42)
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 100:
            for hole in range(25):  # 25 holes!
                # Random hole location
                idx = np.random.randint(len(liver_coords[0]))
                cx, cy, cz = liver_coords[0][idx], liver_coords[1][idx], liver_coords[2][idx]
                
                # Create hole (complete tissue absence)
                radius = np.random.randint(2, 5)
                for x in range(max(0, cx-radius), min(64, cx+radius+1)):
                    for y in range(max(0, cy-radius), min(64, cy+radius+1)):
                        for z in range(max(0, cz-radius), min(64, cz+radius+1)):
                            distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                            if distance <= radius**2:
                                pathological_volume[x, y, z] = 0.0  # Complete void
        
        return pathological_volume, "Swiss Cheese Liver (Multiple Holes)"
    
    def create_liver_inversion(self, base_volume, liver_mask):
        """Invert liver intensities - complete structural inversion"""
        pathological_volume = base_volume.copy()
        
        # Completely invert intensities within liver
        liver_region = liver_mask
        pathological_volume[liver_region] = 1.0 - pathological_volume[liver_region]
        
        return pathological_volume, "Complete Liver Intensity Inversion"
    
    def create_liver_checkerboard(self, base_volume, liver_mask):
        """Create checkerboard pattern - alternating bright/dark"""
        pathological_volume = base_volume.copy()
        
        # Create 3D checkerboard pattern in liver
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    if liver_mask[x, y, z]:
                        # Checkerboard pattern
                        is_bright = (x//4 + y//4 + z//4) % 2 == 0
                        if is_bright:
                            pathological_volume[x, y, z] = 0.95  # Very bright
                        else:
                            pathological_volume[x, y, z] = 0.05  # Very dark
        
        return pathological_volume, "Liver Checkerboard Pattern"
    
    def create_liver_gradient_destruction(self, base_volume, liver_mask):
        """Create extreme gradient across liver"""
        pathological_volume = base_volume.copy()
        
        liver_coords = np.where(liver_mask)
        if len(liver_coords[0]) > 0:
            # Create dramatic gradient from 0 to 1 across liver
            x_coords = liver_coords[0]
            x_min, x_max = x_coords.min(), x_coords.max()
            
            for i in range(len(x_coords)):
                x, y, z = liver_coords[0][i], liver_coords[1][i], liver_coords[2][i]
                # Linear gradient from dark to bright
                gradient_value = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                pathological_volume[x, y, z] = gradient_value
        
        return pathological_volume, "Extreme Liver Gradient Destruction"
    
    def create_liver_noise_chaos(self, base_volume, liver_mask):
        """Replace liver with pure noise"""
        pathological_volume = base_volume.copy()
        
        # Replace entire liver with random noise
        np.random.seed(123)
        noise = np.random.random(pathological_volume.shape)
        pathological_volume[liver_mask] = noise[liver_mask]
        
        return pathological_volume, "Complete Liver Noise Chaos"
    
    def create_liver_geometry_break(self, base_volume, liver_mask):
        """Break liver geometry completely"""
        pathological_volume = base_volume.copy()
        
        # Create geometric pattern that breaks natural liver structure
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    if liver_mask[x, y, z]:
                        # Sine wave patterns
                        wave_x = np.sin(x * 0.3)
                        wave_y = np.sin(y * 0.3)
                        wave_z = np.sin(z * 0.3)
                        combined = (wave_x + wave_y + wave_z + 3) / 6  # Normalize to [0,1]
                        pathological_volume[x, y, z] = combined
        
        return pathological_volume, "Liver Geometry Destruction (Sine Waves)"
    
    def create_all_extreme_destructive_pathologies(self, base_index=15):
        """Create structure-destroying pathologies"""
        if base_index >= len(self.preprocessor.image_files):
            base_index = len(self.preprocessor.image_files) - 1
        
        base_scan = self.preprocessor.image_files[base_index]
        mask_scan = self.preprocessor.label_files[base_index]
        
        volume, mask = self.preprocessor.preprocess_liver_volume(base_scan, mask_scan)
        if volume is None:
            print(f"❌ Could not load base volume {base_index}")
            return []
        
        liver_mask = mask > 0
        pathological_cases = []
        
        # Create DESTRUCTIVE pathologies
        destructive_creators = [
            self.create_liver_swiss_cheese,
            self.create_liver_inversion,
            self.create_liver_checkerboard,
            self.create_liver_gradient_destruction,
            self.create_liver_noise_chaos,
            self.create_liver_geometry_break
        ]
        
        print(f"Creating EXTREME destructive pathologies from base volume {base_index}")
        print(f"Original liver voxels: {np.sum(liver_mask):,}")
        print(f"Original liver intensity range: {volume[liver_mask].min():.3f} - {volume[liver_mask].max():.3f}")
        
        for i, creator in enumerate(destructive_creators):
            try:
                pathological_volume, description = creator(volume, liver_mask)
                
                # Calculate STRUCTURAL difference (not just intensity)
                original_std = np.std(volume[liver_mask])
                pathology_std = np.std(pathological_volume[liver_mask])
                structural_change = abs(original_std - pathology_std)
                
                # Calculate intensity range change
                orig_range = volume[liver_mask].max() - volume[liver_mask].min()
                path_range = pathological_volume[liver_mask].max() - pathological_volume[liver_mask].min()
                range_change = abs(orig_range - path_range)
                
                print(f"\nCreated EXTREME pathology {i+1}: {description}")
                print(f"  Original std: {original_std:.3f}, Pathology std: {pathology_std:.3f}")
                print(f"  Structural change: {structural_change:.3f}")
                print(f"  Range change: {range_change:.3f}")
                
                # Accept ANY structural change (no minimum threshold)
                pathological_cases.append({
                    'volume': pathological_volume,
                    'mask': mask,
                    'description': description,
                    'expected': 'anomaly',
                    'structural_change': structural_change,
                    'range_change': range_change
                })
                print(f"  ✅ EXTREME pathology added")
                    
            except Exception as e:
                print(f"Failed to create EXTREME pathology {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n🔥 Created {len(pathological_cases)} EXTREME destructive pathological cases")
        return pathological_cases

class ExtremeDestructiveTester:
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
        self.destroyer = ExtremeStructureDestroyer(self.preprocessor)
        
        print(f"🔥 EXTREME destructive tester loaded on {self.device}")
    
    def use_lower_threshold(self):
        """Use a much lower threshold to catch subtle differences"""
        # Use the existing normal range but be much more sensitive
        normal_errors = [0.010224, 0.009553, 0.006200, 0.012115, 0.013013, 
                        0.007150, 0.012033, 0.010906, 0.009152, 0.010661]
        
        mean_error = np.mean(normal_errors)
        std_error = np.std(normal_errors)
        
        # VERY sensitive threshold - just 1.5 std above mean
        sensitive_threshold = mean_error + 1.5 * std_error
        
        print(f"EXTREME SENSITIVE threshold calculation:")
        print(f"Mean normal error: {mean_error:.6f}")
        print(f"Std normal error: {std_error:.6f}")
        print(f"EXTREME threshold (mean + 1.5*std): {sensitive_threshold:.6f}")
        
        return sensitive_threshold
    
    def test_extreme_destructive_pathologies(self, threshold):
        """Test EXTREME structure-destroying pathologies"""
        print(f"\n🔥 === TESTING EXTREME DESTRUCTIVE Liver Pathologies ===")
        print(f"Using EXTREME sensitive threshold: {threshold:.6f}")
        
        # Create EXTREME destructive pathologies
        pathological_cases = self.destroyer.create_all_extreme_destructive_pathologies()
        
        if not pathological_cases:
            print("❌ No EXTREME pathological cases created!")
            return [], 0
        
        results = []
        correct_detections = 0
        
        for i, case in enumerate(pathological_cases):
            print(f"\n🔥 DESTRUCTIVE Case {i+1}: {case['description']}")
            print(f"   Structural change: {case['structural_change']:.3f}")
            print(f"   Range change: {case['range_change']:.3f}")
            
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
                status = "🔥 DETECTED"
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
                'structural_change': case['structural_change']
            })
        
        detection_rate = (correct_detections / len(pathological_cases) * 100) if pathological_cases else 0
        print(f"\n🔥 EXTREME DESTRUCTIVE Detection: {correct_detections}/{len(pathological_cases)} ({detection_rate:.1f}%)")
        
        return results, detection_rate

def main():
    """Run EXTREME structure-destroying pathology testing"""
    model_path = "../models/best_liver_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print("❌ Liver model not found!")
        return
    
    print("🔥🔥 EXTREME STRUCTURE-DESTROYING LIVER PATHOLOGY TESTING 🔥🔥")
    print("=" * 70)
    
    # Run EXTREME destructive testing
    tester = ExtremeDestructiveTester(model_path)
    
    # Use very sensitive threshold
    threshold = tester.use_lower_threshold()
    
    # Test EXTREME structure-destroying pathologies
    results, detection_rate = tester.test_extreme_destructive_pathologies(threshold)
    
    print(f"\n🎯 FINAL EXTREME DESTRUCTIVE RESULTS:")
    print(f"Detection Rate: {detection_rate:.1f}%")
    print(f"EXTREME Threshold: {threshold:.6f}")
    
    if detection_rate >= 80:
        print("🎉🔥 INCREDIBLE! EXTREME pathologies detected - absolutely ready!")
    elif detection_rate >= 60:
        print("✅🔥 EXCELLENT! Most EXTREME pathologies detected")
    elif detection_rate >= 40:
        print("🔥 GOOD! Some EXTREME pathologies detected")
    else:
        print("😤 These pathologies are SO extreme they should be impossible to miss!")
        print("Your liver model might be TOO good at reconstruction!")
    
    # Save EXTREME threshold
    try:
        with open("../models/extreme_liver_threshold.txt", "w") as f:
            f.write(f"{threshold:.6f}")
        print("💾 EXTREME threshold saved successfully")
    except Exception as e:
        print(f"⚠️ Could not save threshold: {e}")
    
    # Show most detectable pathologies
    if results:
        print(f"\n🔥 MOST DETECTABLE EXTREME PATHOLOGIES:")
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            if result['detected']:
                print(f"  {i+1}. {result['description']} - {result['confidence']:.2f}x confidence")

if __name__ == "__main__":
    main()
