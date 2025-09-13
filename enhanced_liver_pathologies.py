import torch
import numpy as np
from pathlib import Path
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

class LiverPathologyCreator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def create_hepatocellular_carcinoma(self, base_volume, liver_mask):
        """Create large hepatocellular carcinoma (HCC) - most common liver cancer"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 200:
            # Large HCC tumor (20x20x20 voxels - 3-4cm equivalent)
            center_idx = len(liver_coords[0]) // 4
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Create large hypervascular mass
            for x in range(max(0, cx-10), min(64, cx+11)):
                for y in range(max(0, cy-10), min(64, cy+11)):
                    for z in range(max(0, cz-10), min(64, cz+11)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 100:  # Large sphere
                            # Very bright hypervascular lesion
                            pathological_volume[x, y, z] = min(1.0, pathological_volume[x, y, z] + 0.7)
                            
        return pathological_volume, "Large Hepatocellular Carcinoma (HCC)"
    
    def create_liver_metastases(self, base_volume, liver_mask):
        """Create multiple liver metastases - very common"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 300:
            # Multiple metastatic lesions (5-6 lesions)
            np.random.seed(42)  # Reproducible
            
            for lesion in range(6):
                # Random locations in liver
                idx = np.random.randint(len(liver_coords[0]))
                cx, cy, cz = liver_coords[0][idx], liver_coords[1][idx], liver_coords[2][idx]
                
                # Variable lesion sizes (metastases vary in size)
                radius = np.random.randint(4, 8)  # 4-8 voxel radius
                
                for x in range(max(0, cx-radius), min(64, cx+radius+1)):
                    for y in range(max(0, cy-radius), min(64, cy+radius+1)):
                        for z in range(max(0, cz-radius), min(64, cz+radius+1)):
                            distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                            if distance <= radius**2:
                                # Hypodense metastases (darker than normal liver)
                                pathological_volume[x, y, z] = max(0.0, pathological_volume[x, y, z] - 0.6)
                                
        return pathological_volume, "Multiple Liver Metastases"
    
    def create_liver_hemangioma(self, base_volume, liver_mask):
        """Create liver hemangioma - benign but distinctive"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 150:
            # Large hemangioma (bright, well-defined)
            center_idx = len(liver_coords[0]) // 2
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Large bright lesion with characteristic pattern
            for x in range(max(0, cx-7), min(64, cx+8)):
                for y in range(max(0, cy-7), min(64, cy+8)):
                    for z in range(max(0, cz-7), min(64, cz+8)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 49:  # Sphere radius 7
                            # Very bright vascular lesion
                            pathological_volume[x, y, z] = min(1.0, pathological_volume[x, y, z] + 0.8)
                            
        return pathological_volume, "Large Liver Hemangioma"
    
    def create_liver_cysts(self, base_volume, liver_mask):
        """Create liver cysts - fluid-filled, very hypodense"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 200:
            # Multiple large cysts
            for cyst in range(3):
                idx = len(liver_coords[0]) // (cyst + 2)
                cx, cy, cz = liver_coords[0][idx], liver_coords[1][idx], liver_coords[2][idx]
                
                # Large cyst (very hypodense - near zero intensity)
                radius = 6 + cyst * 2  # Variable sizes
                for x in range(max(0, cx-radius), min(64, cx+radius+1)):
                    for y in range(max(0, cy-radius), min(64, cy+radius+1)):
                        for z in range(max(0, cz-radius), min(64, cz+radius+1)):
                            distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                            if distance <= radius**2:
                                # Very hypodense (fluid density)
                                pathological_volume[x, y, z] = 0.05  # Near water density
                                
        return pathological_volume, "Multiple Large Liver Cysts"
    
    def create_liver_abscess(self, base_volume, liver_mask):
        """Create liver abscess - infected fluid collection"""
        pathological_volume = base_volume.copy()
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 150:
            # Large abscess with thick wall
            center_idx = len(liver_coords[0]) // 3
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Create abscess with rim enhancement
            for x in range(max(0, cx-8), min(64, cx+9)):
                for y in range(max(0, cy-8), min(64, cy+9)):
                    for z in range(max(0, cz-8), min(64, cz+9)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        
                        if distance <= 36:  # Inner cavity (hypodense)
                            pathological_volume[x, y, z] = 0.1  # Fluid/pus
                        elif distance <= 64:  # Rim (hyperenhancing)
                            pathological_volume[x, y, z] = min(1.0, pathological_volume[x, y, z] + 0.6)
                            
        return pathological_volume, "Large Liver Abscess with Rim Enhancement"
    
    def create_all_liver_pathologies(self, base_index=10):
        """Create comprehensive liver pathological test cases"""
        if base_index >= len(self.preprocessor.image_files):
            return []
        
        base_scan = self.preprocessor.image_files[base_index]
        mask_scan = self.preprocessor.label_files[base_index]
        
        volume, mask = self.preprocessor.preprocess_liver_volume(base_scan, mask_scan)
        if volume is None:
            return []
        
        liver_mask = mask > 0
        pathological_cases = []
        
        # Create each type of liver pathology
        pathology_creators = [
            self.create_hepatocellular_carcinoma,
            self.create_liver_metastases,
            self.create_liver_hemangioma,
            self.create_liver_cysts,
            self.create_liver_abscess
        ]
        
        for creator in pathology_creators:
            try:
                pathological_volume, description = creator(volume, liver_mask)
                pathological_cases.append({
                    'volume': pathological_volume,
                    'mask': mask,
                    'description': description,
                    'expected': 'anomaly'
                })
            except Exception as e:
                print(f"Failed to create liver pathology: {e}")
                continue
        
        return pathological_cases

class EnhancedLiverAnomalyTester:
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
        self.pathology_creator = LiverPathologyCreator(self.preprocessor)
        
        print(f"✅ Enhanced liver tester loaded on {self.device}")
    
    def calculate_liver_threshold_enhanced(self):
        """Calculate threshold with more normal samples"""
        print("Calculating enhanced liver threshold...")
        
        normal_errors = []
        for i in range(min(20, len(self.preprocessor.image_files))):
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
                
            except:
                continue
        
        if normal_errors:
            mean_error = np.mean(normal_errors)
            std_error = np.std(normal_errors)
            
            # Enhanced thresholds for different sensitivity levels
            thresholds = {
                'conservative': mean_error + 4.0 * std_error,  # Very specific
                'balanced': mean_error + 2.5 * std_error,      # Balanced  
                'sensitive': mean_error + 2.0 * std_error,     # More sensitive
            }
            
            print(f"Enhanced liver error analysis:")
            print(f"Normal errors range: {min(normal_errors):.6f} - {max(normal_errors):.6f}")
            print(f"Mean: {mean_error:.6f}, Std: {std_error:.6f}")
            print(f"Conservative threshold: {thresholds['conservative']:.6f}")
            print(f"Balanced threshold: {thresholds['balanced']:.6f}")
            print(f"Sensitive threshold: {thresholds['sensitive']:.6f}")
            
            return thresholds, normal_errors
        
        return {'balanced': 0.020}, []
    
    def test_enhanced_liver_pathologies(self, threshold):
        """Test enhanced liver pathological cases"""
        print(f"\n=== Testing Enhanced Liver Pathologies ===")
        print(f"Using threshold: {threshold:.6f}")
        
        # Create realistic liver pathologies
        pathological_cases = self.pathology_creator.create_all_liver_pathologies()
        
        results = []
        correct_detections = 0
        
        for i, case in enumerate(pathological_cases):
            print(f"\n🫘 Liver Case {i+1}: {case['description']}")
            
            # Test pathological case
            liver_mask = case['mask'] > 0
            liver_volume = case['volume'].copy()
            liver_volume[~liver_mask] = 0
            
            volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            is_anomaly = error > threshold
            confidence = error / threshold
            
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
                'confidence': confidence
            })
        
        detection_rate = correct_detections / len(pathological_cases) * 100 if pathological_cases else 0
        print(f"\n📊 Enhanced Liver Pathology Detection: {correct_detections}/{len(pathological_cases)} ({detection_rate:.1f}%)")
        
        return results, detection_rate
    
    def comprehensive_liver_evaluation(self):
        """Run comprehensive liver evaluation with multiple thresholds"""
        print("=" * 60)
        print("🫘 COMPREHENSIVE ENHANCED LIVER ANOMALY EVALUATION")
        print("=" * 60)
        
        # Calculate enhanced thresholds
        thresholds, normal_errors = self.calculate_liver_threshold_enhanced()
        
        best_threshold = None
        best_score = 0
        
        for threshold_name, threshold_value in thresholds.items():
            print(f"\n{'='*50}")
            print(f"🎯 TESTING WITH {threshold_name.upper()} THRESHOLD: {threshold_value:.6f}")
            print(f"{'='*50}")
            
            # Test pathological cases
            results, detection_rate = self.test_enhanced_liver_pathologies(threshold_value)
            
            # Test normal cases for false positives
            normal_fps = 0
            normal_tested = 0
            for i in range(min(5, len(self.preprocessor.image_files))):
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
                    
                    if error > threshold_value:
                        normal_fps += 1
                    normal_tested += 1
                    
                except:
                    continue
            
            fp_rate = (normal_fps / normal_tested * 100) if normal_tested > 0 else 0
            
            # Overall score (balance detection rate and low false positives)
            score = detection_rate * (1 - fp_rate/100)
            
            print(f"\n📊 RESULTS FOR {threshold_name.upper()} THRESHOLD:")
            print(f"   Pathology Detection Rate: {detection_rate:.1f}%")
            print(f"   False Positive Rate: {fp_rate:.1f}%")
            print(f"   Overall Score: {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold_value
        
        print(f"\n🏆 RECOMMENDED LIVER THRESHOLD: {best_threshold:.6f}")
        print(f"🎯 EXPECTED LIVER PERFORMANCE: {best_score:.1f}% overall accuracy")
        
        return best_threshold

def main():
    """Run enhanced liver pathology testing"""
    model_path = "../models/best_liver_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print("❌ Liver model not found!")
        return
    
    # Run enhanced testing
    tester = EnhancedLiverAnomalyTester(model_path)
    recommended_threshold = tester.comprehensive_liver_evaluation()
    
    print(f"\n✅ Enhanced liver testing completed!")
    print(f"💡 Use threshold {recommended_threshold:.6f} for liver hackathon demo")
    
    # Save recommended threshold
    with open("../models/recommended_liver_threshold.txt", "w") as f:
        f.write(f"{recommended_threshold:.6f}")
    
    print("🔧 Liver threshold saved to models/recommended_liver_threshold.txt")

if __name__ == "__main__":
    main()
