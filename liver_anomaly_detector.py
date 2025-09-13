import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from liver_3d_model import Liver3DAutoencoder
from liver_preprocessing import LiverDataPreprocessor

class Liver3DAnomalyDetector:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained liver model
        self.model = Liver3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
        
        print(f"✅ Loaded liver model from {model_path}")
        print(f"Using device: {self.device}")
        print(f"Best validation loss from training: {checkpoint.get('val_loss', 'unknown'):.6f}")
    
    def calculate_liver_threshold(self):
        """Calculate anomaly threshold from normal liver volumes"""
        print("Calculating liver anomaly threshold from normal volumes...")
        
        normal_errors = []
        
        # Test on first 15 training volumes (known normal liver)
        test_count = min(15, len(self.preprocessor.image_files))
        
        for i in range(test_count):
            volume_path = self.preprocessor.image_files[i]
            mask_path = self.preprocessor.label_files[i]
            
            try:
                # Preprocess liver volume
                volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                # Create liver-only volume (normal liver tissue)
                liver_mask = mask > 0
                liver_volume = volume.copy()
                liver_volume[~liver_mask] = 0  # Zero out non-liver regions
                
                volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                # Calculate reconstruction error
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    normal_errors.append(error)
                
                print(f"  Liver volume {i+1}: reconstruction error = {error:.6f}")
                
            except Exception as e:
                print(f"  Volume {i+1}: Error - {e}")
                continue
        
        if normal_errors:
            mean_error = np.mean(normal_errors)
            std_error = np.std(normal_errors)
            
            # Liver threshold - more conservative due to organ complexity
            threshold = mean_error + 3.0 * std_error  # 3 sigma for liver
            
            print(f"\nLiver Threshold Calculation:")
            print(f"Normal reconstruction errors: {[f'{e:.6f}' for e in normal_errors[:5]]}...")
            print(f"Mean error: {mean_error:.6f}")
            print(f"Std error: {std_error:.6f}")
            print(f"Threshold (mean + 3*std): {threshold:.6f}")
            
            return threshold, normal_errors
        else:
            print("❌ Could not calculate threshold - no valid normal volumes")
            return 0.025, []  # Conservative default for liver
    
    def detect_anomaly_from_training_file(self, volume_idx, threshold=None):
        """Test detection on a training liver volume"""
        if threshold is None:
            threshold = 0.025  # Default liver threshold
        
        if volume_idx >= len(self.preprocessor.image_files):
            print(f"❌ Index {volume_idx} out of range")
            return None
        
        volume_path = self.preprocessor.image_files[volume_idx]
        mask_path = self.preprocessor.label_files[volume_idx]
        
        print(f"Analyzing liver volume {volume_idx}: {volume_path.name}")
        
        try:
            # Use same preprocessing as training
            volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
            if volume is None:
                return None
            
            # Create liver-only volume
            liver_mask = mask > 0
            liver_volume = volume.copy()
            liver_volume[~liver_mask] = 0
            
            volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            # Model inference
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            # Anomaly detection
            is_anomaly = reconstruction_error > threshold
            confidence = reconstruction_error / threshold if threshold > 0 else 0
            
            result = {
                'volume_idx': volume_idx,
                'file_path': str(volume_path),
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'reconstruction_error': reconstruction_error,
                'threshold': threshold,
                'liver_voxels': np.sum(liver_mask),
                'original_shape': volume.shape
            }
            
            print(f"Reconstruction Error: {reconstruction_error:.6f}")
            print(f"Threshold: {threshold:.6f}")
            print(f"Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ NORMAL LIVER'}")
            print(f"Confidence: {confidence:.2f}x threshold")
            print(f"Liver voxels: {np.sum(liver_mask):,}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing liver volume {volume_idx}: {e}")
            return None
    
    def create_liver_pathology_test(self, base_idx=10):
        """Create synthetic liver pathology for testing"""
        if base_idx >= len(self.preprocessor.image_files):
            return None, None
        
        volume_path = self.preprocessor.image_files[base_idx]
        mask_path = self.preprocessor.label_files[base_idx]
        
        # Load normal liver
        volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
        if volume is None:
            return None, None
        
        # Create synthetic liver lesion (hepatocellular carcinoma simulation)
        pathological_volume = volume.copy()
        liver_mask = mask > 0
        liver_coords = np.where(liver_mask)
        
        if len(liver_coords[0]) > 100:
            # Large hypervascular lesion (bright on arterial phase)
            center_idx = len(liver_coords[0]) // 3
            cx, cy, cz = liver_coords[0][center_idx], liver_coords[1][center_idx], liver_coords[2][center_idx]
            
            # Large spherical lesion (15x15x15 voxels - typical HCC size)
            for x in range(max(0, cx-8), min(64, cx+8)):
                for y in range(max(0, cy-8), min(64, cy+8)):
                    for z in range(max(0, cz-8), min(64, cz+8)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 64:  # Sphere radius 8
                            # Make it hyperdense (like contrast-enhanced lesion)
                            pathological_volume[x, y, z] = min(1.0, pathological_volume[x, y, z] + 0.4)
        
        return pathological_volume, mask

def test_liver_anomaly_detection():
    """Test liver anomaly detection system"""
    print("=== Testing 3D Liver Anomaly Detection System ===")
    
    # Check if trained model exists
    model_path = "../models/best_liver_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print(f"❌ Liver model not found: {model_path}")
        print("Please complete liver training first!")
        return
    
    # Initialize detector
    detector = Liver3DAnomalyDetector(model_path)
    
    # Calculate liver-specific threshold
    threshold, normal_errors = detector.calculate_liver_threshold()
    
    # Test on normal liver volumes
    print(f"\n=== Testing Normal Liver Volumes ===")
    normal_results = []
    test_indices = [0, 5, 10]  # Test 3 different normal volumes
    
    for idx in test_indices:
        print(f"\n📋 Testing normal liver volume {idx}:")
        result = detector.detect_anomaly_from_training_file(idx, threshold)
        if result:
            normal_results.append(result)
    
    # Test synthetic liver pathology
    print(f"\n=== Testing Synthetic Liver Pathology ===")
    pathological_volume, mask = detector.create_liver_pathology_test(base_idx=8)
    
    if pathological_volume is not None:
        print("🩺 Testing synthetic hepatocellular carcinoma (HCC):")
        
        # Test synthetic pathology
        liver_mask = mask > 0
        masked_pathology = pathological_volume.copy()
        masked_pathology[~liver_mask] = 0
        
        volume_tensor = torch.FloatTensor(masked_pathology[np.newaxis, np.newaxis, ...]).to(detector.device)
        
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        is_anomaly = error > threshold
        confidence = error / threshold
        
        print(f"Synthetic HCC reconstruction error: {error:.6f}")
        print(f"Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '❌ MISSED'}")
        print(f"Confidence: {confidence:.2f}x threshold")
    
    # Performance summary
    if normal_results:
        false_positives = sum(1 for r in normal_results if r['is_anomaly'])
        fp_rate = false_positives / len(normal_results) * 100
        
        print(f"\n📊 LIVER ANOMALY DETECTION PERFORMANCE:")
        print(f"Normal volumes tested: {len(normal_results)}")
        print(f"False positives: {false_positives}")
        print(f"False positive rate: {fp_rate:.1f}%")
        print(f"Liver-optimized threshold: {threshold:.6f}")
        
        if fp_rate <= 10:
            print("✅ EXCELLENT: Low false positive rate - ready for liver demo!")
        elif fp_rate <= 25:
            print("✅ GOOD: Acceptable performance for liver anomaly detection")
        else:
            print("⚠️  Consider threshold adjustment for better specificity")

if __name__ == "__main__":
    test_liver_anomaly_detection()
