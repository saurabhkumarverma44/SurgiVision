import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model_regularized import LiverAutoencoderRegularized
import copy

class LiverDatasetAugmented(Dataset):
    def __init__(self, preprocessor, indices, augment=True):
        self.preprocessor = preprocessor
        self.indices = indices
        self.augment = augment
        
        # Filter valid indices
        self.valid_indices = [i for i in indices if i < len(preprocessor.image_files)]
        print(f"Dataset created with {len(self.valid_indices)} samples, augmentation: {augment}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def augment_volume(self, volume):
        """Apply data augmentation to prevent overfitting"""
        if not self.augment:
            return volume
            
        augmented = volume.copy()
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            # Simple rotation simulation by shifting
            shift = int(angle / 5 * 2)  # Convert to voxel shift
            if shift != 0:
                if shift > 0:
                    augmented = np.roll(augmented, shift, axis=0)
                else:
                    augmented = np.roll(augmented, shift, axis=0)
        
        # Random intensity variation
        if np.random.random() > 0.5:
            intensity_factor = np.random.uniform(0.9, 1.1)
            augmented = np.clip(augmented * intensity_factor, 0, 1)
        
        # Random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        # Random gamma correction
        if np.random.random() > 0.6:
            gamma = np.random.uniform(0.8, 1.2)
            augmented = np.power(augmented, gamma)
        
        return augmented
    
    def __getitem__(self, idx):
        volume_idx = self.valid_indices[idx]
        volume_path = self.preprocessor.image_files[volume_idx]
        mask_path = self.preprocessor.label_files[volume_idx]
        
        try:
            volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
            if volume is None:
                # Return dummy data if preprocessing fails
                volume = np.random.rand(64, 64, 64) * 0.3
                mask = np.ones((64, 64, 64)) * 0.5
            
            # Apply augmentation
            volume = self.augment_volume(volume)
            
            # Create liver-only volume
            liver_mask = mask > 0
            liver_volume = volume.copy()
            liver_volume[~liver_mask] = 0
            
            volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, ...])
            return volume_tensor
            
        except Exception as e:
            print(f"Error with volume {volume_idx}: {e}")
            # Return dummy data
            dummy_volume = np.random.rand(64, 64, 64) * 0.3
            return torch.FloatTensor(dummy_volume[np.newaxis, ...])

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Restoring best weights from validation loss: {self.best_loss:.6f}")
            return True
        return False

class RegularizedLiverTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Advanced optimization with regularization
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(  # AdamW has built-in weight decay
            model.parameters(), 
            lr=0.0005,  # Lower learning rate
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler (FIXED - removed verbose)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=1e-6)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, volumes in enumerate(self.train_loader):
            volumes = volumes.to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed = self.model(volumes)
            
            # Main reconstruction loss
            recon_loss = self.criterion(reconstructed, volumes)
            
            # L1 regularization on bottleneck (sparsity)
            l1_reg = 0
            for param in self.model.bottleneck.parameters():
                l1_reg += torch.sum(torch.abs(param))
            
            # Combined loss
            total_loss = recon_loss + 1e-5 * l1_reg  # Small L1 penalty
            
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if batch_idx % 20 == 0:  # More frequent updates for 104 samples
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}: Loss = {total_loss.item():.6f}")
        
        return epoch_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
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
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return val_loss / num_batches if num_batches > 0 else 0
    
    def train_with_early_stopping(self, max_epochs=100):
        print(f"🚀 Starting regularized training for max {max_epochs} epochs with early stopping...")
        print(f"📊 Training samples: {len(self.train_loader)}")
        print(f"📊 Validation samples: {len(self.val_loader)}")
        start_time = time.time()
        
        for epoch in range(max_epochs):
            print(f"\n🔄 Epoch {epoch+1}/{max_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Check if LR was reduced
            if current_lr < old_lr:
                print(f"📉 Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")
            
            print(f"📊 Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"🎯 Best validation loss: {self.early_stopping.best_loss:.6f}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                models_dir = Path("../models")
                models_dir.mkdir(exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, f'../models/liver_checkpoint_epoch_{epoch+1}.pth')
                
                print(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        training_time = time.time() - start_time
        final_epoch = len(self.train_losses)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Epochs trained: {final_epoch}")
        print(f"⏱️ Training time: {training_time/60:.1f} minutes")
        print(f"🎯 Best validation loss: {self.early_stopping.best_loss:.6f}")
        
        # Save final model
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.early_stopping.best_loss,
        }, '../models/liver_regularized_final.pth')
        
        print(f"💾 Final model saved: ../models/liver_regularized_final.pth")
        
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self):
        """Plot training curves to visualize overfitting"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.7)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss (Regularized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, self.learning_rates, 'g-', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/regularized_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training curves saved: ../results/regularized_training_curves.png")

def main():
    print("🫘 REGULARIZED LIVER MODEL TRAINING WITH EARLY STOPPING")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_root = "C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver"
    preprocessor = LiverDataPreprocessor(data_root)
    
    if len(preprocessor.image_files) == 0:
        print("❌ No data found!")
        return
    
    print(f"📊 Total volumes: {len(preprocessor.image_files)}")
    
    # Create proper train/validation split (80/20)
    total_indices = list(range(len(preprocessor.image_files)))
    np.random.seed(42)  # Reproducible split
    np.random.shuffle(total_indices)
    
    split_point = int(0.8 * len(total_indices))
    train_indices = total_indices[:split_point]
    val_indices = total_indices[split_point:]
    
    print(f"📈 Training samples: {len(train_indices)}")
    print(f"📉 Validation samples: {len(val_indices)}")
    
    # Create datasets with augmentation
    train_dataset = LiverDatasetAugmented(preprocessor, train_indices, augment=True)
    val_dataset = LiverDatasetAugmented(preprocessor, val_indices, augment=False)
    
    # Create data loaders
    batch_size = 1  # Keep for RTX 4050
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create regularized model
    model = LiverAutoencoderRegularized(latent_dim=256, dropout_rate=0.3)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = RegularizedLiverTrainer(model, train_loader, val_loader, device)
    
    # Train with early stopping
    train_losses, val_losses = trainer.train_with_early_stopping(max_epochs=100)
    
    # Plot results
    trainer.plot_training_curves()
    
    print(f"\n🎯 REGULARIZED MODEL TRAINING COMPLETED!")
    print(f"📁 Final model: ../models/liver_regularized_final.pth")
    print(f"📊 Training curves: ../results/regularized_training_curves.png")
    print(f"🔄 Ready for evaluation on unseen data!")

if __name__ == "__main__":
    main()
