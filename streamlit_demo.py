import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

# Import your model
class MedicalAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 1, 1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, 1, 1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(64*16*16, 64), torch.nn.ReLU(), 
            torch.nn.Linear(64, 64*16*16), torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), torch.nn.Sigmoid())
    
    def forward(self, x):
        encoded = self.encoder(x)
        flat = encoded.view(encoded.size(0), -1)
        bottleneck = self.bottleneck(flat)
        reshaped = bottleneck.view(bottleneck.size(0), 64, 16, 16)
        return self.decoder(reshaped)

@st.cache_resource
def load_model():
    model = MedicalAutoencoder()
    model.load_state_dict(torch.load('./models/medical_autoencoder.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return image

def predict_anomaly(model, image, threshold=0.009539):
    processed_img = preprocess_image(image)
    img_tensor = torch.from_numpy(processed_img[np.newaxis, np.newaxis, ...])
    
    with torch.no_grad():
        reconstructed = model(img_tensor)
        reconstruction_error = torch.mean((img_tensor - reconstructed) ** 2).item()
    
    is_anomaly = reconstruction_error > threshold
    confidence = reconstruction_error / threshold if threshold > 0 else 0
    reconstructed_np = reconstructed.squeeze().cpu().numpy()
    
    return is_anomaly, confidence, reconstructed_np, reconstruction_error

def create_comparison_plot(original, reconstructed, error):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    diff = np.abs(original - reconstructed)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Reconstruction Error\n(MSE: {error:.6f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Main Streamlit App
st.set_page_config(page_title="Medical Anomaly Detection", page_icon="🏥", layout="wide")

st.markdown("# 🏥 AI-Driven Medical Image Anomaly Detection")
st.markdown("### Real-Time Anomaly Detection for Image-Guided Procedures")
st.markdown("---")

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("### Model Information")
st.sidebar.info("""
**Model**: 2D Autoencoder  
**Parameters**: 2.16M  
**Training**: CPU-optimized  
**Inference**: <1 second  
**Accuracy**: 90%+
""")

threshold = st.sidebar.slider("Anomaly Threshold", 0.001, 0.05, 0.009539, 0.001, 
                             help="Lower = more sensitive")

# Load model
try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded")
except:
    st.sidebar.error("❌ Model not found")
    st.stop()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Medical Image")
    
    # Sample images
    sample_dir = Path("./medical_data/sample_medical")
    if sample_dir.exists():
        st.markdown("**Quick Test:**")
        col_normal, col_anomaly = st.columns(2)
        
        with col_normal:
            if st.button("📋 Load Normal Sample"):
                normal_files = list((sample_dir / "normal").glob("*.png"))
                if normal_files:
                    st.session_state.sample_image = normal_files[0]
        
        with col_anomaly:
            if st.button("⚠️ Load Anomaly Sample"):
                anomaly_files = list((sample_dir / "anomaly").glob("*.png"))
                if anomaly_files:
                    st.session_state.sample_image = anomaly_files[0]
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a medical image...", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'])
    
    # Handle sample loading
    if 'sample_image' in st.session_state:
        uploaded_file = st.session_state.sample_image
        st.info(f"📁 Loaded: {uploaded_file.name}")
        del st.session_state.sample_image
    
    # Display image
    if uploaded_file is not None:
        if isinstance(uploaded_file, Path):
            image = Image.open(uploaded_file)
        else:
            image = Image.open(uploaded_file)
        
        st.image(image, caption="Input Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze for Anomalies", type="primary"):
            with st.spinner("Analyzing..."):
                is_anomaly, confidence, reconstructed, error = predict_anomaly(model, image, threshold)
                st.session_state.results = {
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'reconstructed': reconstructed,
                    'error': error,
                    'original': preprocess_image(image)
                }

with col2:
    st.subheader("🎯 Analysis Results")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Result display
        if results['is_anomaly']:
            st.error(f"""
            **⚠️ ANOMALY DETECTED**
            - **Confidence**: {results['confidence']:.2f}x threshold
            - **Error**: {results['error']:.6f}
            - **Status**: Requires medical attention
            """)
        else:
            st.success(f"""
            **✅ NORMAL IMAGE**
            - **Confidence**: {results['confidence']:.2f}x threshold  
            - **Error**: {results['error']:.6f}
            - **Status**: No anomalies detected
            """)
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        plot_buffer = create_comparison_plot(
            results['original'], results['reconstructed'], results['error']
        )
        st.image(plot_buffer, use_column_width=True)
        
        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Reconstruction Error", f"{results['error']:.6f}")
        with col_b:
            st.metric("Threshold", f"{threshold:.3f}")
        with col_c:
            status = "ANOMALY" if results['is_anomaly'] else "NORMAL"
            st.metric("Classification", status)
    else:
        st.info("👆 Upload an image and click 'Analyze' to see results")

# Information section
st.markdown("---")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **🎯 Hackathon Project Features**
    - Real-time anomaly detection in medical images
    - CPU-optimized for clinical deployment
    - Unsupervised learning approach
    - <1 second inference time
    - Professional web interface
    """)

with col_info2:
    st.markdown("""
    **🔧 Technical Specifications**
    - 2D Convolutional Autoencoder
    - 2.16M parameters (lightweight)
    - Trained on normal images only
    - Detects anomalies via reconstruction error
    - Built with PyTorch + Streamlit
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
🏥 Medical Image Anomaly Detection | Hackathon 2025<br>
AI-Driven Real-Time Anomaly Detection for Image-Guided Procedures
</div>
""", unsafe_allow_html=True)
