# Phase 2: Model Development - Fixed version
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import time

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, image_size=(128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            image = np.zeros(self.image_size, dtype=np.uint8)
        
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        image = image[np.newaxis, ...]
        image = torch.from_numpy(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        
        return image

class MedicalAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super(MedicalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        flattened = encoded.view(batch_size, -1)
        bottleneck_out = self.bottleneck(flattened)
        reshaped = bottleneck_out.view(batch_size, 64, 16, 16)
        decoded = self.decoder(reshaped)
        return decoded

class AnomalyDetector:
    def __init__(self, model_save_path="./models/medical_autoencoder.pth"):
        self.device = torch.device("cpu")
        self.model = MedicalAutoencoder().to(self.device)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(exist_ok=True)
        
        self.batch_size = 2
        self.learning_rate = 0.001
        self.num_epochs = 8
        
        print(f"Initialized MedicalAutoencoder on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, normal_images_path, anomaly_images_path=None):
        print("Preparing data...")
        
        normal_path = Path(normal_images_path)
        normal_files = list(normal_path.glob("*.png"))
        
        if not normal_files:
            print(f"No PNG files found in {normal_path}")
            return 0, 0
        
        print(f"Found {len(normal_files)} normal images")
        
        all_files = normal_files.copy()
        all_labels = ['normal'] * len(normal_files)
        
        if anomaly_images_path and Path(anomaly_images_path).exists():
            anomaly_path = Path(anomaly_images_path)
            anomaly_files = list(anomaly_path.glob("*.png"))
            if anomaly_files:
                all_files.extend(anomaly_files)
                all_labels.extend(['anomaly'] * len(anomaly_files))
                print(f"Found {len(anomaly_files)} anomaly images")
        
        print(f"Total dataset: {len(all_files)} images")
        
        if len(all_files) > 4:
            train_files, val_files, train_labels, val_labels = train_test_split(
                all_files, all_labels, test_size=0.3, random_state=42
            )
        else:
            train_files = val_files = all_files
            train_labels = val_labels = all_labels
        
        normal_train_files = [f for f, l in zip(train_files, train_labels) if l == 'normal']
        
        print(f"Training on {len(normal_train_files)} normal images")
        
        train_dataset = MedicalImageDataset(normal_train_files)
        val_dataset = MedicalImageDataset(val_files, val_labels)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return len(normal_train_files), len(val_files)
    
    def train(self):
        print("\nTraining autoencoder...")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_losses = []
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_idx, images in enumerate(self.train_loader):
                images = images.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx+1}: Loss = {loss.item():.6f}")
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.6f}")
            
            if avg_loss < 0.01:
                print("Loss converged early!")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': train_losses,
            'training_time': training_time
        }, self.model_save_path)
        
        print(f"Model saved to {self.model_save_path}")
        return train_losses
    
    def evaluate(self):
        print("\nEvaluating model...")
        
        self.model.eval()
        reconstruction_errors = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                errors = torch.mean((images - reconstructed) ** 2, dim=(1, 2, 3))
                reconstruction_errors.extend(errors.cpu().numpy())
                true_labels.extend(labels)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        normal_errors = reconstruction_errors[np.array(true_labels) == 'normal']
        if len(normal_errors) > 0:
            threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)
        else:
            threshold = np.median(reconstruction_errors)
        
        print(f"Anomaly threshold: {threshold:.6f}")
        print(f"Avg reconstruction error: {np.mean(reconstruction_errors):.6f}")
        
        anomaly_count = sum(1 for error in reconstruction_errors if error > threshold)
        print(f"Detected {anomaly_count}/{len(reconstruction_errors)} as anomalies")
        
        return threshold, reconstruction_errors

if __name__ == "__main__":
    print("=== PHASE 2: MODEL DEVELOPMENT ===")
    
    detector = AnomalyDetector()
    
    normal_path = "./medical_data/sample_medical/normal"
    anomaly_path = "./medical_data/sample_medical/anomaly"
    
    print(f"Using data from: {normal_path}")
    
    try:
        train_count, val_count = detector.prepare_data(normal_path, anomaly_path)
        
        if train_count > 0:
            print(f"Starting training with {train_count} images...")
            train_losses = detector.train()
            
            threshold, errors = detector.evaluate()
            
            print("\n=== PHASE 2 COMPLETED SUCCESSFULLY ===")
            print(f"Model trained on {train_count} normal images")
            print(f"Anomaly detection threshold: {threshold:.6f}")
            print(f"Model ready for demo!")
            print(f"Saved to: ./models/medical_autoencoder.pth")
            
        else:
            print("No training data found!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
