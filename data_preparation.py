# Phase 1: Data Preparation and Exploration
# File: data_preparation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import requests
import zipfile
import gdown

class MedicalDataPreprocessor:
    def __init__(self, data_dir="./medical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_covid_xray_dataset(self):
        """Download COVID-19 chest X-ray dataset - perfect for hackathon"""
        print("Downloading COVID-19 Chest X-ray dataset...")
        
        # Create directories
        (self.data_dir / "covid_xray").mkdir(exist_ok=True)
        (self.data_dir / "covid_xray" / "normal").mkdir(exist_ok=True)
        (self.data_dir / "covid_xray" / "covid").mkdir(exist_ok=True)
        (self.data_dir / "covid_xray" / "pneumonia").mkdir(exist_ok=True)
        
        # Sample URLs - in real scenario, you'd download from Kaggle or official sources
        sample_data_info = {
            "normal": 100,    # 100 normal chest X-rays
            "covid": 50,      # 50 COVID-19 cases  
            "pneumonia": 75   # 75 pneumonia cases
        }
        
        print("Dataset info:")
        for category, count in sample_data_info.items():
            print(f"- {category}: {count} images")
            
        # Create sample dataset structure file
        dataset_info = pd.DataFrame({
            'category': ['normal', 'covid', 'pneumonia'],
            'count': [100, 50, 75],
            'description': [
                'Normal chest X-rays',
                'COVID-19 positive cases',
                'Pneumonia cases'
            ]
        })
        
        dataset_info.to_csv(self.data_dir / "dataset_info.csv", index=False)
        print(f"Dataset structure saved to {self.data_dir / 'dataset_info.csv'}")
        
        return sample_data_info
    
    def process_brain_tumor_2d(self, brain_data_path):
        """Convert existing 3D brain tumor data to 2D slices for CPU processing"""
        print("Processing brain tumor data to 2D slices...")
        
        # Create output directory
        output_dir = self.data_dir / "brain_tumor_2d"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "masks").mkdir(exist_ok=True)
        
        # If brain_data_path exists, process it
        if os.path.exists(brain_data_path):
            images_path = Path(brain_data_path) / "imagesTr_small"
            labels_path = Path(brain_data_path) / "labelsTr_small"
            
            if images_path.exists() and labels_path.exists():
                image_files = sorted(list(images_path.glob("*.nii*")))
                label_files = sorted(list(labels_path.glob("*.nii*")))
                
                slice_count = 0
                for img_file, label_file in zip(image_files, label_files):
                    try:
                        # Load 3D volumes
                        img_data = nib.load(str(img_file)).get_fdata()
                        label_data = nib.load(str(label_file)).get_fdata()
                        
                        # Extract middle slices (most informative)
                        mid_slice = img_data.shape[2] // 2
                        start_slice = max(0, mid_slice - 5)
                        end_slice = min(img_data.shape[2], mid_slice + 5)
                        
                        for slice_idx in range(start_slice, end_slice):
                            # Get 2D slice from first channel
                            img_slice = img_data[:, :, slice_idx, 0]  # First channel
                            label_slice = label_data[:, :, slice_idx]
                            
                            # Skip empty slices
                            if np.max(img_slice) > 0:
                                # Normalize to 0-255
                                img_normalized = ((img_slice - np.min(img_slice)) / 
                                                (np.max(img_slice) - np.min(img_slice) + 1e-8) * 255).astype(np.uint8)
                                
                                label_normalized = (label_slice > 0).astype(np.uint8) * 255
                                
                                # Save as PNG
                                cv2.imwrite(str(output_dir / "images" / f"slice_{slice_count:04d}.png"), img_normalized)
                                cv2.imwrite(str(output_dir / "masks" / f"slice_{slice_count:04d}.png"), label_normalized)
                                
                                slice_count += 1
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
                
                print(f"Converted {slice_count} 2D slices from brain tumor data")
                return slice_count
        
        # If no data exists, create synthetic sample data for demo
        return self.create_sample_medical_data()
    
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
        """Explore and visualize the prepared dataset"""
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
            
        if (dataset_path / "covid").exists():
            covid_files = list((dataset_path / "covid").glob("*.png"))
            image_files.extend(covid_files)
            labels.extend(['covid'] * len(covid_files))
        
        print(f"Total images found: {len(image_files)}")
        
        # Create visualization
        if image_files:
            self.visualize_samples(image_files[:8], labels[:8])
            
        # Create dataset statistics
        label_counts = pd.Series(labels).value_counts()
        print("\nDataset distribution:")
        print(label_counts)
        
        return image_files, labels
    
    def visualize_samples(self, image_files, labels):
        """Visualize sample images"""
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, (img_file, label) in enumerate(zip(image_files[:8], labels[:8])):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {self.data_dir / 'sample_visualization.png'}")

# Main execution for Phase 1
if __name__ == "__main__":
    print("=== PHASE 1: DATA PREPARATION ===")
    
    # Initialize preprocessor
    preprocessor = MedicalDataPreprocessor()
    
    # Option 1: Download COVID dataset (if internet available)
    try:
        covid_info = preprocessor.download_covid_xray_dataset()
    except:
        print("Could not download COVID dataset, proceeding with local data...")
    
    # Option 2: Process existing brain tumor data or create sample data
    brain_data_path = "../data/Task01_BrainTumour/Task01_BrainTumour"
    total_images = preprocessor.process_brain_tumor_2d(brain_data_path)
    
    # Option 3: Create additional sample data
    if total_images < 50:  # If we don't have enough data
        sample_count = preprocessor.create_sample_medical_data()
        print(f"Created {sample_count} additional sample images")
    
    # Explore the prepared dataset
    sample_dataset_path = preprocessor.data_dir / "sample_medical"
    if sample_dataset_path.exists():
        image_files, labels = preprocessor.explore_dataset(sample_dataset_path)
        print(f"\nPhase 1 Complete: Prepared {len(image_files)} images")
    
    print("=== PHASE 1 COMPLETED ===")
