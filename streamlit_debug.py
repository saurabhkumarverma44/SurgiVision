import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io
import traceback

# Model class (same as before)
class MedicalAutoencoder(torch.nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super(MedicalAutoencoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )
        
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, latent_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(latent_dim, 64 * 16 * 16),
            torch.nn.ReLU(inplace=True),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        flattened = encoded.view(batch_size, -1)
        bottleneck_out = self.bottleneck(flattened)
        reshaped = bottleneck_out.view(batch_size, 64, 16, 16)
        decoded = self.decoder(reshaped)
        return decoded

@st.cache_resource
def load_model():
    try:
        model = MedicalAutoencoder()
        checkpoint = torch.load('./models/medical_autoencoder.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.write("✅ Model loaded successfully!")
        return model, True, "Model loaded successfully"
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        st.write(f"❌ {error_msg}")
        return None, False, error_msg

def preprocess_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        return image, True, "Preprocessing successful"
    except Exception as e:
        return None, False, f"Preprocessing failed: {str(e)}"

def predict_anomaly(model, image, threshold=0.009539):
    try:
        st.write("🔄 Starting prediction...")
        
        # Preprocess
        processed_img, success, msg = preprocess_image(image)
        if not success:
            return None, None, None, None, f"Preprocessing error: {msg}"
        
        st.write("✅ Image preprocessed")
        
        # Create tensor
        img_tensor = torch.from_numpy(processed_img[np.newaxis, np.newaxis, ...])
        st.write(f"✅ Tensor created: shape {img_tensor.shape}")
        
        # Model inference
        with torch.no_grad():
            reconstructed = model(img_tensor)
            reconstruction_error = torch.mean((img_tensor - reconstructed) ** 2).item()
        
        st.write(f"✅ Model inference completed. Error: {reconstruction_error:.6f}")
        
        # Calculate results
        is_anomaly = reconstruction_error > threshold
        confidence = reconstruction_error / threshold if threshold > 0 else 0
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        
        st.write(f"✅ Results calculated. Anomaly: {is_anomaly}, Confidence: {confidence:.2f}")
        
        return is_anomaly, confidence, reconstructed_np, reconstruction_error, "Success"
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        return None, None, None, None, error_msg

# Streamlit App
st.set_page_config(page_title="Medical Anomaly Detection - DEBUG", page_icon="🏥", layout="wide")

st.markdown("# 🏥 Medical Anomaly Detection - DEBUG VERSION")
st.markdown("---")

# Load model with debug info
st.subheader("🔧 Debug Information")
model, model_loaded, load_msg = load_model()
st.write(f"Model Status: {load_msg}")

if model_loaded:
    st.success("✅ Model is ready for testing")
else:
    st.error("❌ Model failed to load - check model file exists")
    st.stop()

st.markdown("---")

# Simple test interface
st.subheader("📤 Test Interface")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Load Sample Images:**")
    
    if st.button("📋 Load Normal Sample"):
        normal_files = list(Path("./medical_data/sample_medical/normal").glob("*.png"))
        if normal_files:
            st.session_state.test_image = normal_files[0]
            st.session_state.image_type = "normal"
            st.success(f"Loaded: {normal_files[0].name}")
    
    if st.button("⚠️ Load Anomaly Sample"):
        anomaly_files = list(Path("./medical_data/sample_medical/anomaly").glob("*.png"))
        if anomaly_files:
            st.session_state.test_image = anomaly_files[0]
            st.session_state.image_type = "anomaly"
            st.success(f"Loaded: {anomaly_files[0].name}")

with col2:
    st.markdown("**Analysis:**")
    threshold = st.slider("Threshold", 0.001, 0.05, 0.009539, 0.001)
    
    if st.button("🔍 ANALYZE (DEBUG)", type="primary"):
        if 'test_image' in st.session_state:
            st.write("🚀 Starting analysis...")
            
            # Load and display image
            image = Image.open(st.session_state.test_image)
            st.image(image, caption=f"Testing {st.session_state.image_type} image", width=200)
            
            # Run prediction with full debug output
            is_anomaly, confidence, reconstructed, error, status = predict_anomaly(model, image, threshold)
            
            if is_anomaly is not None:
                # Display results
                if is_anomaly:
                    st.error(f"🚨 ANOMALY DETECTED (Error: {error:.6f}, Confidence: {confidence:.2f}x)")
                else:
                    st.success(f"✅ NORMAL (Error: {error:.6f}, Confidence: {confidence:.2f}x)")
            else:
                st.error(f"❌ Analysis failed: {status}")
        else:
            st.warning("Please load a sample image first!")

# Additional debug info
st.markdown("---")
st.subheader("🔍 System Check")

# Check file paths
normal_path = Path("./medical_data/sample_medical/normal")
anomaly_path = Path("./medical_data/sample_medical/anomaly")
model_path = Path("./models/medical_autoencoder.pth")

st.write(f"Normal samples exist: {normal_path.exists()} ({len(list(normal_path.glob('*.png'))) if normal_path.exists() else 0} files)")
st.write(f"Anomaly samples exist: {anomaly_path.exists()} ({len(list(anomaly_path.glob('*.png'))) if anomaly_path.exists() else 0} files)")
st.write(f"Model file exists: {model_path.exists()}")

if model_path.exists():
    st.write(f"Model file size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
