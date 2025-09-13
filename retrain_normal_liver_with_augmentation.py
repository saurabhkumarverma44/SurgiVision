import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from scipy import ndimage
import random
import nibabel as nib
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

class LiverAugmentation:
    """Heavy augmentation for 13 normal liver volumes"""
    
    def __init__(self):
        pass
    
    def augment_volume(self, volume, mask):
        """Apply heavy augmentation to create new training samples"""
        
        # Random rotation (±20 degrees)
        angle_xy = random.uniform(-20, 20)
        angle_xz = random.uniform(-10, 10)
        angle_yz = random.uniform(-10, 10)
        
        volume_aug = ndimage.rotate(volume, angle_xy, axes=(0,1), reshape=False, order=1)
        mask_aug = ndimage.rotate(mask, angle_xy, axes=(0,1), reshape=False, order=0)
        
        volume_aug = ndimage.rotate(volume_aug, angle_xz, axes=(0,2), reshape=False, order=1)
        mask_aug = ndimage.rotate(mask_aug, angle_xz, axes=(0,2), reshape=False, order=0)
        
        # Random flip
        if random.random() > 0.5:
            volume_aug = np.flip(volume_aug, axis=0)
            mask_aug = np.flip(mask_aug, axis=0)
        
        if random.random() > 0.5:
            volume_aug = np.flip(volume_aug, axis=1)
            mask_aug = np.flip(mask_aug, axis=1)
        
        # Random scale (simulate different liver sizes)
        scale_factor = random.uniform(0.8, 1.2)
        if scale_factor != 1.0:
            zoom_factors = [scale_factor] * 3
            volume_aug = ndimage.zoom(volume_aug, zoom_factors, order=1)
            mask_aug = ndimage.zoom(mask_aug, zoom_factors, order=0)
            
            # Resize back to 64x64x64
            current_shape = volume_aug.shape
            final_zoom = [64/current_shape[i] for i in range(3)]
            volume_aug = ndimage.zoom(volume_aug, final_zoom, order=1)
            mask_aug = ndimage.zoom(mask_aug, final_zoom, order=0)
        
        # Intensity augmentation
        intensity_shift = random.uniform(-0.1, 0.1)
        intensity_scale = random.uniform(0.9, 1.1)
        volume_aug = (volume_aug + intensity_shift) * intensity_scale
        volume_aug = np.clip(volume_aug, 0, 1)
        
        # Add slight noise
        noise_factor = random.uniform(0.01, 0.03)
        volume_aug += np.random.normal(0, noise_factor, volume_aug.shape)
        volume_aug = np.clip(volume_aug, 0, 1)
        
        # Elastic deformation (simulate different patient anatomies)
        if random.random() > 0.7:  # 30% chance
            # Simple elastic deformation
            displacement = np.random.random((3, 8, 8, 8)) * 4 - 2  # Random displacement field
            displacement_full = ndimage.zoom(displacement, [1, 8, 8, 8], order=1)
            
            # Apply small deformation
            indices = np.mgrid[0:64, 0:64, 0:64]
            for i in range(3):
                indices[i] = indices[i] + displacement_full[i] * 0.1  # Small deformation
            
            try:
                volume_aug = ndimage.map_coordinates(volume_aug, indices, order=1, cval=0)
                mask_aug = ndimage.map_coordinates(mask_aug, indices, order=0, cval=0)
            except:
                pass  # If deformation fails, use original
        
        return volume_aug, mask_aug

class NormalLiverDataset(Dataset):
    """Dataset with ONLY normal liver volumes + heavy augmentation"""
    
    def __init__(self, preprocessor, normal_indices, train=True, augmentations_per_volume=25):
        self.preprocessor = preprocessor
        self.normal_indices = normal_indices
        self.augmentation = LiverAugmentation()
        self.train = train
        self.augmentations_per_volume = augmentations_per_volume
        
        print(f"🫀 Creating dataset from {len(normal_indices)} normal liver volumes")
        print(f"📈 Augmentations per volume: {augmentations_per_volume}")
        
        # Load and augment all normal volumes
        self.data = []
        self.load_and_augment_data()
        
        # 80-20 split
        split_idx = int(0.8 * len(self.data))
        if train:
            self.samples = self.data[:split_idx]
        else:
            self.samples = self.data[split_idx:]
        
        print(f"{'Training' if train else 'Validation'} samples: {len(self.samples)}")
    
    def load_and_augment_data(self):
        """Load and create augmented dataset"""
        for idx in self.normal_indices:
            volume_path = self.preprocessor.image_files[idx]
            mask_path = self.preprocessor.label_files[idx]
            
            print(f"📋 Processing normal volume {idx}: {volume_path.name}")
            
            # Load original
            volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
            
            if volume is None:
                continue
            
            # Add original
            self.data.append({'volume': volume, 'mask': mask})
            
            # Create augmentations
            for aug_idx in range(self.augmentations_per_volume):
                try:
                    aug_volume, aug_mask = self.augmentation.augment_volume(volume, mask)
                    self.data.append({'volume': aug_volume, 'mask': aug_mask})
                except Exception as e:
                    print(f"  ⚠️ Augmentation {aug_idx} failed: {e}")
                    continue
        
        print(f"✅ Total dataset size: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        volume = sample['volume']
        mask = sample['mask']
        
        # Apply liver mask (only normal liver tissue)
        liver_mask = mask > 0
        masked_volume = volume.copy()
        masked_volume[~liver_mask] = 0
        
        volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, ...])
        return volume_tensor

def get_normal_liver_indices():
    """Get indices of pure normal liver volumes (no tumors)"""
    print("🔍 Finding pure normal liver volumes...")
    
    preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
    normal_indices = []
    
    for i in range(len(preprocessor.image_files)):
        try:
            mask_path = preprocessor.label_files[i]
            mask = nib.load(mask_path).get_fdata()
            unique_labels = np.unique(mask)
            
            # Check if it has liver (1) but NO tumors (2)
            has_liver = 1 in unique_labels
            has_tumor = 2 in unique_labels
            
            if has_liver and not has_tumor:
                normal_indices.append(i)
                print(f"✅ Normal volume {i}: {preprocessor.image_files[i].name}")
                
        except Exception as e:
            continue
    
    print(f"📊 Found {len(normal_indices)} pure normal liver volumes")
    return normal_indices, preprocessor

class NormalLiverTrainer:
    """Trainer for normal liver only with augmented data"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training configuration
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train for one epoch on normal liver only"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, volumes in enumerate(self.train_loader):
            volumes = volumes.to(self.device)

            self.optimizer.zero_grad()
            reconstructed = self.model(volumes)
            loss = self.criterion(reconstructed, volumes)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx+1}: Loss = {loss.item():.6f}")

        return epoch_loss / num_batches

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for volumes in self.val_loader:
                volumes = volumes.to(self.device)
                reconstructed = self.model(volumes)
                loss = self.criterion(reconstructed, volumes)
                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def train(self, num_epochs=80):
        """Full training loop"""
        print(f"🫀 Training NORMAL LIVER ANOMALY DETECTOR for {num_epochs} epochs")
        start_time = time.time()

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, '../models/best_liver_3d_autoencoder_NORMAL_TRAINED.pth')
                print(f"✅ New best NORMAL LIVER model saved (val_loss: {val_loss:.6f})")

            # Early stopping
            if current_lr < 1e-6:
                print("Learning rate too small, stopping training")
                break

        training_time = time.time() - start_time
        print(f"\n✅ NORMAL LIVER TRAINING completed in {training_time:.1f} seconds")
        print(f"🎯 Best validation loss: {best_val_loss:.6f}")

        return self.train_losses, self.val_losses

def main():
    """Retrain liver model on ONLY normal liver volumes"""
    print("🫀 RETRAINING LIVER MODEL - NORMAL LIVER ANOMALY DETECTION")
    print("="*80)

    # Get normal liver indices
    normal_indices, preprocessor = get_normal_liver_indices()
    
    if len(normal_indices) < 8:
        print(f"❌ Not enough normal liver volumes! Found only {len(normal_indices)}")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create heavily augmented datasets
    train_dataset = NormalLiverDataset(preprocessor, normal_indices, train=True, augmentations_per_volume=30)
    val_dataset = NormalLiverDataset(preprocessor, normal_indices, train=False, augmentations_per_volume=30)

    # Create data loaders
    batch_size = 2 if device.type == 'cuda' else 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create NEW model (fresh start)
    model = Liver3DAutoencoder()

    # Create trainer
    trainer = NormalLiverTrainer(model, train_loader, val_loader, device)

    # Train model on NORMAL LIVER ONLY
    train_losses, val_losses = trainer.train(num_epochs=120)

    print("\n🎉 NORMAL LIVER ANOMALY DETECTOR TRAINING COMPLETED!")
    print("📁 New model saved: ../models/best_liver_3d_autoencoder_NORMAL_TRAINED.pth")
    
    print("\n🎯 This model should now:")
    print("  ✅ Show normal liver volumes as NORMAL")
    print("  🚨 Show tumor volumes as ANOMALY") 
    print("  🚨 Show synthetic pathology as ANOMALY")

if __name__ == "__main__":
    main()
