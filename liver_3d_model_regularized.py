import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LiverAutoencoderRegularized(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256, dropout_rate=0.3):  # Reduced latent dim
        super(LiverAutoencoderRegularized, self).__init__()
        
        # 3D Encoder with regularization
        self.encoder = nn.Sequential(
            # Input: 1×64×64×64 → 16×32×32×32
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.5),  # Light dropout in encoder
            nn.MaxPool3d(2),
            
            # 16×32×32×32 → 32×16×16×16  
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.7),
            nn.MaxPool3d(2),
            
            # 32×16×16×16 → 64×8×8×8
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),  # Higher dropout deeper
            nn.MaxPool3d(2),
        )
        
        # Bottleneck with heavy regularization
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8 * 8, latent_dim),  # Smaller bottleneck: 32768 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, latent_dim // 2),  # Even smaller: 256 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim // 2, latent_dim),  # Back up: 128 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, 64 * 8 * 8 * 8),  # 256 → 32768
            nn.ReLU(inplace=True),
        )
        
        # 3D Decoder
        self.decoder = nn.Sequential(
            # 64×8×8×8 → 32×16×16×16
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 32×16×16×16 → 16×32×32×32
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 16×32×32×32 → 1×64×64×64
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Add noise for regularization during training
        if self.training:
            noise = torch.randn_like(x) * 0.01  # Small noise
            x = x + noise
            
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        
        flattened = encoded.view(batch_size, -1)
        bottleneck_out = self.bottleneck(flattened)
        reshaped = bottleneck_out.view(batch_size, 64, 8, 8, 8)
        
        decoded = self.decoder(reshaped)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        features = self.bottleneck[0](flattened)
        return features

def test_regularized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiverAutoencoderRegularized().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Regularized model parameters: {total_params:,}")
    
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Regularized model test successful: {output.shape}")

if __name__ == "__main__":
    test_regularized_model()
