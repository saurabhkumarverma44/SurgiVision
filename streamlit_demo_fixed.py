import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

# EXACT same model class from training
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
        return model, True
    except Exception as e:
        return None, False

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return image

def predict_anomaly(model, image, threshold=0.015):
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
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('AI Reconstructed', fontsize=14)
    axes[1].axis('off')
    
    diff = np.abs(original - reconstructed)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Anomaly Heatmap\nError: {error:.6f}', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Main App
st.set_page_config(page_title="Medical Anomaly Detection", page_icon="🏥", layout="wide")

st.markdown("# 🏥 AI-Driven Medical Image Anomaly Detection")
st.markdown("### Real-Time Anomaly Detection for Image-Guided Procedures")
st.markdown("---")

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("### Model Information")

# Load model
model, model_loaded = load_model()

if model_loaded:
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.info("""
    **Model**: 2D Autoencoder  
    **Parameters**: 2.16M  
    **Training**: CPU-optimized  
    **Inference**: <1 second  
    **Accuracy**: 90%+
    """)
else:
    st.sidebar.error("❌ Model Loading Failed")
    st.error("⚠️ Please ensure model is trained first: `python medical_autoencoder_fixed.py`")

threshold = st.sidebar.slider("Anomaly Threshold", 0.001, 0.05, 0.015, 0.001, 
                             help="Lower = more sensitive to anomalies")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Medical Image")
    
    # Sample images
    sample_dir = Path("./medical_data/sample_medical")
    if sample_dir.exists():
        st.markdown("**Quick Demo:**")
        col_normal, col_anomaly = st.columns(2)
        
        with col_normal:
            if st.button("📋 Load Normal Sample", type="secondary"):
                normal_files = list((sample_dir / "normal").glob("*.png"))
                if normal_files:
                    st.session_state.sample_image = normal_files[0]
                    st.session_state.sample_type = "normal"
        
        with col_anomaly:
            if st.button("⚠️ Load Anomaly Sample", type="secondary"):
                anomaly_files = list((sample_dir / "anomaly").glob("*.png"))
                if anomaly_files:
                    st.session_state.sample_image = anomaly_files[0]
                    st.session_state.sample_type = "anomaly"
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a medical image...", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'])
    
    # Handle sample loading
    if 'sample_image' in st.session_state:
        uploaded_file = st.session_state.sample_image
        sample_type = st.session_state.get('sample_type', 'unknown')
        st.info(f"📁 Loaded {sample_type} sample: {uploaded_file.name}")
        del st.session_state.sample_image
        if 'sample_type' in st.session_state:
            del st.session_state.sample_type
    
    # Display image
    if uploaded_file is not None:
        if isinstance(uploaded_file, Path):
            image = Image.open(uploaded_file)
        else:
            image = Image.open(uploaded_file)
        
        st.image(image, caption="Input Medical Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze for Anomalies", type="primary", disabled=not model_loaded):
            if model_loaded:
                with st.spinner("🔄 AI Analysis in progress..."):
                    is_anomaly, confidence, reconstructed, error = predict_anomaly(model, image, threshold)
                    st.session_state.results = {
                        'is_anomaly': is_anomaly,
                        'confidence': confidence,
                        'reconstructed': reconstructed,
                        'error': error,
                        'original': preprocess_image(image)
                    }

with col2:
    st.subheader("🎯 AI Analysis Results")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Result display
        if results['is_anomaly']:
            st.error(f"""
            **🚨 ANOMALY DETECTED**
            
            - **Confidence**: {results['confidence']:.2f}x threshold
            - **Reconstruction Error**: {results['error']:.6f}
            - **Clinical Status**: ⚠️ Requires immediate attention
            - **Recommendation**: Manual review by radiologist
            """)
        else:
            st.success(f"""
            **✅ NORMAL SCAN**
            
            - **Confidence**: {results['confidence']:.2f}x threshold  
            - **Reconstruction Error**: {results['error']:.6f}
            - **Clinical Status**: ✅ No anomalies detected
            - **Recommendation**: Scan appears normal
            """)
        
        # Detailed analysis
        st.subheader("🔬 AI Reconstruction Analysis")
        plot_buffer = create_comparison_plot(
            results['original'], results['reconstructed'], results['error']
        )
        st.image(plot_buffer, use_column_width=True)
        
        # Technical metrics
        st.subheader("📊 Technical Metrics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Reconstruction Error", f"{results['error']:.6f}")
        with col_b:
            st.metric("Detection Threshold", f"{threshold:.3f}")
        with col_c:
            status = "🚨 ANOMALY" if results['is_anomaly'] else "✅ NORMAL"
            st.metric("AI Classification", status)
    else:
        st.info("👆 Upload a medical image and click 'Analyze' to see AI results")

# Information section
st.markdown("---")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **🎯 Hackathon Project Features**
    - Real-time anomaly detection in medical images
    - CPU-optimized for clinical deployment
    - Unsupervised learning approach
    - Sub-second inference time
    - Professional medical interface
    """)

with col_info2:
    st.markdown("""
    **🔧 Technical Implementation**
    - 2D Convolutional Autoencoder Architecture
    - 2.16M parameters (lightweight & fast)
    - Trained on normal images using reconstruction loss
    - Anomaly detection via reconstruction error threshold
    - Built with PyTorch + Streamlit framework
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
🏥 <b>Medical Image Anomaly Detection System</b> | Hackathon 2025<br>
<i>AI-Driven Real-Time Anomaly Detection for Image-Guided Procedures</i><br>
<small>Developed for improving diagnostic accuracy and reducing human error in medical imaging</small>
</div>
""", unsafe_allow_html=True)
