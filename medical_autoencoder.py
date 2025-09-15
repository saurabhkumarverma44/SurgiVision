# Phase 2: Model Development - Medical Anomaly Detection Autoencoder
# File: medical_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import time

class MedicalImageDataset(Dataset):
    """Dataset class for medical images"""
    def __init__(self, image_paths, labels=None, transform=None, image_size=(128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = str(self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            image = np.zeros(self.image_size, dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension
        image = image[np.newaxis, ...]  # Shape: (1, H, W)
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        
        return image

class MedicalAutoencoder(nn.Module):
    """Lightweight 2D Autoencoder for anomaly detection - CPU optimized"""
    def __init__(self, input_channels=1, latent_dim=128):
        super(MedicalAutoencoder, self).__init__()
        
        # Encoder - much smaller than UNet++
        self.encoder = nn.Sequential(
            # Input: 1x128x128
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x64x64
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x32x32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x16x16
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x8x8
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Reshape to 256x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1x128x128
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        batch_size = encoded.size(0)
        flattened = encoded.view(batch_size, -1)
        bottleneck_out = self.bottleneck(flattened)
        reshaped = bottleneck_out.view(batch_size, 256, 8, 8)
        
        # Decode
        decoded = self.decoder(reshaped)
        
        return decoded

class AnomalyDetector:
    """Main class for training and using the anomaly detection model"""
    
    def __init__(self, model_save_path="./models/medical_autoencoder.pth"):
        self.device = torch.device("cpu")  # CPU only for hackathon
        self.model = MedicalAutoencoder().to(self.device)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(exist_ok=True)
        
        # Training parameters optimized for CPU
        self.batch_size = 4  # Small batch size for CPU
        self.learning_rate = 0.001
        self.num_epochs = 15  # Reasonable for hackathon timeframe
        
        print(f"Initialized MedicalAutoencoder on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, normal_images_path, anomaly_images_path=None):
        """Prepare training and validation data"""
        print("Preparing data...")
        
        normal_path = Path(normal_images_path)
        normal_files = list(normal_path.glob("*.png")) + list(normal_path.glob("*.jpg"))
        normal_labels = ['normal'] * len(normal_files)
        
        all_files = normal_files
        all_labels = normal_labels
        
        if anomaly_images_path and Path(anomaly_images_path).exists():
            anomaly_path = Path(anomaly_images_path)
            anomaly_files = list(anomaly_path.glob("*.png")) + list(anomaly_path.glob("*.jpg"))
            anomaly_labels = ['anomaly'] * len(anomaly_files)
            
            all_files.extend(anomaly_files)
            all_labels.extend(anomaly_labels)
        
        print(f"Total images: {len(all_files)}")
        print(f"Normal: {normal_labels.count('normal')}, Anomaly: {all_labels.count('anomaly')}")
        
        # Split data
        if len(all_files) > 10:  # Only split if we have enough data
            train_files, val_files, train_labels, val_labels = train_test_split(
                all_files, all_labels, test_size=0.2, random_state=42, 
                stratify=all_labels if len(set(all_labels)) > 1 else None
            )
        else:
            # For small datasets, use all data for training
            train_files, val_files = all_files, all_files[:min(3, len(all_files))]
            train_labels, val_labels = all_labels, all_labels[:min(3, len(all_labels))]
        
        # For anomaly detection, we primarily train on normal images
        normal_train_files = [f for f, l in zip(train_files, train_labels) if l == 'normal']
        
        # Create datasets
        train_dataset = MedicalImageDataset(normal_train_files)  # Only normal images for training
        val_dataset = MedicalImageDataset(val_files, val_labels)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Training samples: {len(normal_train_files)}")
        print(f"Validation samples: {len(val_files)}")
        
        return len(normal_train_files), len(val_files)
    
    def train(self):
        """Train the autoencoder model"""
        print("\n=== TRAINING AUTOENCODER ===")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()  # Reconstruction loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, images in enumerate(self.train_loader):
                images = images.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = self.model(images)
                loss = criterion(reconstructed, images)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Progress update
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            train_losses.append(avg_loss)
            
            print(f'Epoch {epoch+1}/{self.num_epochs} Complete - Average Loss: {avg_loss:.6f}')
            
            # Early stopping if loss is very small
            if avg_loss < 0.001:
                print("Loss converged, stopping training early.")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': train_losses,
            'training_time': training_time
        }, self.model_save_path)
        
        print(f"Model saved to {self.model_save_path}")
        
        return train_losses
    
    def evaluate(self, threshold=None):
        """Evaluate the model and determine anomaly threshold"""
        print("\n=== EVALUATING MODEL ===")
        
        self.model.eval()
        reconstruction_errors = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                
                # Calculate reconstruction error for each image
                errors = torch.mean((images - reconstructed) ** 2, dim=(1, 2, 3))
                reconstruction_errors.extend(errors.cpu().numpy())
                true_labels.extend(labels)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Determine threshold if not provided
        if threshold is None:
            # Use mean + 2*std of normal images as threshold
            normal_errors = reconstruction_errors[np.array(true_labels) == 'normal']
            if len(normal_errors) > 0:
                threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)
            else:
                threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile
        
        # Make predictions
        predictions = ['anomaly' if error > threshold else 'normal' for error in reconstruction_errors]
        
        # Calculate metrics if we have both normal and anomaly samples
        if 'anomaly' in true_labels and 'normal' in true_labels:
            # Convert to binary for metrics
            y_true = [1 if label == 'anomaly' else 0 for label in true_labels]
            y_pred = [1 if pred == 'anomaly' else 0 for pred in predictions]
            
            auc_score = roc_auc_score(y_true, reconstruction_errors)
            print(f"AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            print(classification_report(true_labels, predictions))
        
        print(f"Anomaly threshold: {threshold:.6f}")
        print(f"Average reconstruction error: {np.mean(reconstruction_errors):.6f}")
        
        return threshold, reconstruction_errors, predictions

# Main execution for Phase 2
if __name__ == "__main__":
    print("=== PHASE 2: MODEL DEVELOPMENT ===")
    
    # Initialize anomaly detector
    detector = AnomalyDetector()
    
    # Prepare data - adjust paths based on your Phase 1 output
    normal_path = "./medical_data/sample_medical/normal"
    anomaly_path = "./medical_data/sample_medical/anomaly"
    
    # Alternative: use brain tumor 2D slices if available
    brain_2d_path = "./medical_data/brain_tumor_2d/images"
    if Path(brain_2d_path).exists():
        print("Using brain tumor 2D data...")
        normal_path = brain_2d_path
        anomaly_path = None  # Will create synthetic anomalies
    
    try:
        train_count, val_count = detector.prepare_data(normal_path, anomaly_path)
        
        if train_count > 0:
            # Train the model
            train_losses = detector.train()
            
            # Evaluate the model
            threshold, errors, predictions = detector.evaluate()
            
            print("\n=== PHASE 2 COMPLETED SUCCESSFULLY ===")
            print(f"Model trained on {train_count} normal images")
            print(f"Anomaly detection threshold: {threshold:.6f}")
            print(f"Model saved and ready for demo!")
            
        else:
            print("No training data found. Please run Phase 1 first.")
            
    except Exception as e:
        print(f"Error in Phase 2: {e}")
        print("Make sure Phase 1 completed successfully and data is available.")
