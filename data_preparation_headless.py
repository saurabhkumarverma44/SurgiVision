# Same as data_preparation.py but without plt.show()
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd

class MedicalDataPreprocessor:
    def __init__(self, data_dir="./medical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def create_sample_medical_data(self):
        """Create synthetic medical images for demo purposes"""
        print("Creating sample medical data for demonstration...")
        
        output_dir = self.data_dir / "sample_medical"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "normal").mkdir(exist_ok=True)
        (output_dir / "anomaly").mkdir(exist_ok=True)
        
        np.random.seed(42)  # Reproducible results
        
        # Create normal medical images (simulated)
        normal_count = 50
        anomaly_count = 20
        
        for i in range(normal_count):
            # Simulate normal medical image
            img = np.zeros((256, 256), dtype=np.uint8)
            
            # Add some realistic medical image features
            # Circular organ structure
            center = (128, 128)
            radius = np.random.randint(60, 100)
            cv2.circle(img, center, radius, 150, -1)
            
            # Add texture
            noise = np.random.normal(0, 20, (256, 256))
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            
            # Smooth the image
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            cv2.imwrite(str(output_dir / "normal" / f"normal_{i:03d}.png"), img)
        
        # Create anomalous images
        for i in range(anomaly_count):
            # Start with normal structure
            img = np.zeros((256, 256), dtype=np.uint8)
            center = (128, 128)
            radius = np.random.randint(60, 100)
            cv2.circle(img, center, radius, 150, -1)
            
            # Add anomaly - bright spot (tumor simulation)
            anomaly_center = (np.random.randint(80, 176), np.random.randint(80, 176))
            anomaly_radius = np.random.randint(10, 30)
            cv2.circle(img, anomaly_center, anomaly_radius, 255, -1)
            
            # Add texture and noise
            noise = np.random.normal(0, 20, (256, 256))
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            cv2.imwrite(str(output_dir / "anomaly" / f"anomaly_{i:03d}.png"), img)
        
        print(f"Created {normal_count} normal and {anomaly_count} anomalous sample images")
        return normal_count + anomaly_count
    
    def explore_dataset(self, dataset_path):
        """Explore dataset without hanging visualization"""
        print("\nExploring dataset...")
        dataset_path = Path(dataset_path)
        
        # Get all image files
        image_files = []
        labels = []
        
        if (dataset_path / "normal").exists():
            normal_files = list((dataset_path / "normal").glob("*.png"))
            image_files.extend(normal_files)
            labels.extend(['normal'] * len(normal_files))
            
        if (dataset_path / "anomaly").exists():
            anomaly_files = list((dataset_path / "anomaly").glob("*.png"))
            image_files.extend(anomaly_files)
            labels.extend(['anomaly'] * len(anomaly_files))
        
        print(f"Total images found: {len(image_files)}")
        
        # Create dataset statistics
        label_counts = pd.Series(labels).value_counts()
        print("\nDataset distribution:")
        print(label_counts)
        
        print("✅ Dataset exploration completed!")
        return image_files, labels

# Quick execution
if __name__ == "__main__":
    print("=== PHASE 1: DATA PREPARATION (HEADLESS) ===")
    
    preprocessor = MedicalDataPreprocessor()
    sample_count = preprocessor.create_sample_medical_data()
    
    sample_dataset_path = preprocessor.data_dir / "sample_medical"
    if sample_dataset_path.exists():
        image_files, labels = preprocessor.explore_dataset(sample_dataset_path)
        print(f"\n✅ Phase 1 Complete: Prepared {len(image_files)} images")
    
    print("=== PHASE 1 COMPLETED ===")
