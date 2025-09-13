import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

class LiverDataset(Dataset):
    def __init__(self, preprocessor, train=True, normal_only=True):
        self.preprocessor = preprocessor
        self.normal_only = normal_only
        
        # Check if we have any files
        if len(preprocessor.image_files) == 0:
            raise ValueError(f"No liver image files found! Check data path: {preprocessor.data_root}")
        
        # Use 80% for training, 20% for validation
        n_files = len(preprocessor.image_files)
        split_idx = int(0.8 * n_files)
        
        if train:
            self.image_files = preprocessor.image_files[:split_idx]
            self.label_files = preprocessor.label_files[:split_idx]
        else:
            self.image_files = preprocessor.image_files[split_idx:]
            self.label_files = preprocessor.label_files[split_idx:]
        
        print(f"{'Training' if train else 'Validation'} dataset: {len(self.image_files)} volumes")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No files for {'training' if train else 'validation'} dataset!")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        volume_path = self.image_files[idx]
        mask_path = self.label_files[idx]
        
        # Preprocess volume
        volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
        
        if volume is None:
            # Return dummy data if preprocessing fails
            print(f"⚠️ Preprocessing failed for {volume_path.name}, using dummy data")
            volume = np.random.rand(64, 64, 64) * 0.5  # Random but reasonable data
            mask = np.ones((64, 64, 64)) * 0.5
        
        # Convert to torch tensors
        volume_tensor = torch.FloatTensor(volume[np.newaxis, ...])  # Add channel dim
        
        if self.normal_only:
            # For autoencoder: only return liver regions (normal tissue)
            liver_mask = torch.FloatTensor(mask) > 0
            masked_volume = volume_tensor.clone()
            # Zero out non-liver regions
            masked_volume[0, ~liver_mask] = 0
            return masked_volume
        else:
            # Return full volume and mask
            mask_tensor = torch.FloatTensor(mask[np.newaxis, ...])
            return volume_tensor, mask_tensor

class LiverTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training configuration optimized for RTX 4050 Laptop
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch with RTX 4050 memory management"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, volumes in enumerate(self.train_loader):
            volumes = volumes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(volumes)
            loss = self.criterion(reconstructed, volumes)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # RTX 4050 memory management
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if batch_idx % 30 == 0:  # Progress updates for 201 images
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}: Loss = {loss.item():.6f}")
                
                # Memory monitoring for RTX 4050
                if self.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                    print(f"    GPU Memory: {memory_used:.2f}GB / 6GB")
        
        return epoch_loss / num_batches if num_batches > 0 else 0
    
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
                
                # Memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return val_loss / num_batches if num_batches > 0 else 0
    
    def train(self, num_epochs=25):
        """Full training loop optimized for 201 liver volumes"""
        print(f"Starting liver training for {num_epochs} epochs on {self.device}")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation  
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Create models directory if it doesn't exist
                models_dir = Path("../models")
                models_dir.mkdir(exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, '../models/best_liver_3d_autoencoder.pth')
                print(f"✅ New best liver model saved (val_loss: {val_loss:.6f})")
            
            # Early stopping
            if current_lr < 1e-6:
                print("Learning rate too small, stopping training")
                break
        
        training_time = time.time() - start_time
        print(f"\nLiver training completed in {training_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses

def main():
    """Main liver training function"""
    print("=== 3D Liver Autoencoder Training (RTX 4050 Laptop Optimized) ===")
    
    # Setup device - ensure CUDA is used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"⚠️ CUDA not available, using CPU")
        print("This will be very slow! Please check CUDA installation.")
    
    # Use the correct data path provided by user
    data_root = "C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver"
    print(f"Looking for liver data at: {data_root}")
    
    # Verify data path exists
    if not Path(data_root).exists():
        print(f"❌ Data path does not exist: {data_root}")
        print("Please verify the Task03_Liver folder location.")
        return
    
    try:
        # Initialize preprocessor
        preprocessor = LiverDataPreprocessor(data_root)
        
        if len(preprocessor.image_files) == 0:
            print("❌ No liver images found! Please check:")
            print(f"1. Folder exists: {data_root}")
            print(f"2. imagesTr subfolder exists: {Path(data_root) / 'imagesTr'}")
            print(f"3. .nii.gz files are present")
            return
        
        print(f"✅ Found {len(preprocessor.image_files)} liver images")
        
        # Create datasets
        train_dataset = LiverDataset(preprocessor, train=True, normal_only=True)
        val_dataset = LiverDataset(preprocessor, train=False, normal_only=True)
        
        # Create data loaders - RTX 4050 Laptop optimized
        batch_size = 1  # Safe for 6GB VRAM with liver volumes
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"Batch size: {batch_size} (RTX 4050 Laptop optimized)")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Create model
        model = Liver3DAutoencoder()
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        trainer = LiverTrainer(model, train_loader, val_loader, device)
        
        # Train model
        print("\n🚀 Starting training...")
        train_losses, val_losses = trainer.train(num_epochs=20)
        
        print("\n✅ Liver training completed successfully!")
        print("Model saved to: ../models/best_liver_3d_autoencoder.pth")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
