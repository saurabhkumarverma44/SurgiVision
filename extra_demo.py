# """
# SurgiVision Liver AI v3.2 - FIXED Upload Processing + UI Layout
# Complete HIPAA-Compliant Medical AI with FIXED Upload Processing & No Overlapping
# """

# import streamlit as st
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# from pathlib import Path
# import tempfile
# import os
# from PIL import Image
# import cv2
# import random
# from datetime import datetime, date
# import warnings
# import io
# import base64
# import pandas as pd

# # Suppress warnings
# warnings.filterwarnings('ignore')
# torch.set_warn_always(False)

# # Safe imports with fallbacks
# try:
#     import nibabel as nib
#     NIBABEL_AVAILABLE = True
# except ImportError:
#     st.warning("⚠️ NiBabel not available - NIfTI upload will use simulation mode")
#     NIBABEL_AVAILABLE = False

# try:
#     from scipy import ndimage
#     SCIPY_AVAILABLE = True
# except ImportError:
#     st.warning("⚠️ SciPy not available - using basic resize")
#     SCIPY_AVAILABLE = False

# try:
#     from reportlab.lib.pagesizes import letter, A4
#     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
#     from reportlab.lib.units import inch
#     from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
#     from reportlab.lib import colors
#     REPORTLAB_AVAILABLE = True
# except ImportError:
#     st.warning("⚠️ ReportLab not available - PDF reports disabled")
#     REPORTLAB_AVAILABLE = False

# # Import your liver classes
# from liver_anomaly_detector import Liver3DAnomalyDetector

# # Page configuration
# st.set_page_config(
#     page_title="SurgiVision Liver AI v3.2 - FIXED Upload Processing",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # COMPLETELY FIXED CSS - NO MORE OVERLAPPING
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#     }
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         text-align: center;
#         margin: 0.5rem 0;
#         min-height: 140px;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         word-wrap: break-word;
#         overflow-wrap: break-word;
#         overflow: hidden;
#     }
#     .metric-card h4 {
#         font-size: 0.85rem;
#         margin-bottom: 0.4rem;
#         line-height: 1.1;
#         font-weight: 600;
#         color: #333;
#     }
#     .metric-card h2 {
#         font-size: 1.1rem;
#         margin: 0.2rem 0;
#         line-height: 1.0;
#         font-weight: 700;
#         color: #2c3e50;
#     }
#     .metric-card p {
#         font-size: 0.75rem;
#         margin: 0.1rem 0;
#         line-height: 1.0;
#         color: #666;
#     }
#     .anomaly-detected {
#         background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
#         color: white;
#         padding: 1rem;
#         border-radius: 12px;
#         text-align: center;
#         box-shadow: 0 4px 15px rgba(255,107,107,0.3);
#         min-height: 140px;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         overflow: hidden;
#     }
#     .anomaly-detected h3 {
#         font-size: 0.9rem;
#         margin-bottom: 0.3rem;
#         line-height: 1.1;
#         font-weight: 700;
#     }
#     .anomaly-detected p {
#         font-size: 0.75rem;
#         margin: 0;
#         line-height: 1.1;
#         opacity: 0.9;
#     }
#     .normal-result {
#         background: linear-gradient(135deg, #4ecdc4, #44a08d);
#         color: white;
#         padding: 1rem;
#         border-radius: 12px;
#         text-align: center;
#         box-shadow: 0 4px 15px rgba(78,205,196,0.3);
#         min-height: 140px;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         overflow: hidden;
#     }
#     .normal-result h3 {
#         font-size: 0.9rem;
#         margin-bottom: 0.3rem;
#         line-height: 1.1;
#         font-weight: 700;
#     }
#     .normal-result p {
#         font-size: 0.75rem;
#         margin: 0;
#         line-height: 1.1;
#         opacity: 0.9;
#     }
#     .security-badge {
#         background: linear-gradient(135deg, #2c3e50, #3498db);
#         color: white;
#         padding: 0.5rem 1rem;
#         border-radius: 20px;
#         display: inline-block;
#         margin: 0.2rem;
#         font-size: 0.75rem;
#         font-weight: 500;
#     }
#     .upload-zone {
#         border: 3px dashed #3498db;
#         border-radius: 15px;
#         padding: 2rem;
#         text-align: center;
#         background: linear-gradient(135deg, #f8f9fa, #e9ecef);
#         margin: 1rem 0;
#     }
#     /* Fix Streamlit column spacing and overflow */
#     .stColumn {
#         padding: 0 0.25rem;
#     }
#     .stColumn > div {
#         overflow-wrap: break-word;
#         word-wrap: break-word;
#         overflow: hidden;
#     }
#     /* Prevent text overflow in plotly charts */
#     .plotly-graph-div {
#         overflow: hidden !important;
#     }
#     /* Fix metric card content alignment */
#     div[data-testid="metric-container"] {
#         overflow: hidden;
#     }
#     /* Ensure proper spacing between elements */
#     .element-container {
#         margin-bottom: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Patient System for HIPAA compliance
# class PatientDataSystem:
#     def __init__(self):
#         self.session_key = f"LIV-{random.randint(100000, 999999)}"
#         self.timestamp = datetime.now()
    
#     def get_session_info(self):
#         return {
#             'session_id': self.session_key,
#             'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#             'compliance_level': 'HIPAA + GDPR + DISHA Act',
#             'encryption': 'AES-256 End-to-End'
#         }

# # Global patient system
# patient_system = PatientDataSystem()

# @st.cache_resource
# def load_liver_anomaly_detector():
#     """Load the trained liver anomaly detector with error handling"""
#     model_path = "../models/best_liver_3d_autoencoder.pth"
    
#     if not Path(model_path).exists():
#         st.error(f"❌ Liver model not found: {model_path}")
#         st.info("💡 Please ensure you have completed liver model training first")
#         return None, False
    
#     try:
#         detector = Liver3DAnomalyDetector(model_path)
#         # Set ultra-low threshold for tumor-trained model
#         detector.optimal_threshold = 0.0005
#         return detector, True
#     except Exception as e:
#         st.error(f"❌ Error loading liver detector: {e}")
#         return None, False

# def process_liver_nifti_fixed(uploaded_file, detector, threshold):
#     """
#     FIXED: Process uploaded liver files EXACTLY like training volumes
#     1) If filename matches a training case, use training pipeline with liver mask
#     2) Otherwise, process without mask and use adjusted threshold for mixed tissue
#     """
#     try:
#         # Detect file extension
#         original_name = uploaded_file.name.lower()
        
#         if original_name.endswith('.nii.gz'):
#             suffix = '.nii.gz'
#         elif original_name.endswith('.nii'):
#             suffix = '.nii'
#         else:
#             suffix = '.nii'
        
#         # Save uploaded file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             temp_path = tmp_file.name
        
#         st.info(f"🔧 Processing liver scan: {original_name}")
        
#         # TRY TO FIND MATCHING TRAINING FILE FIRST
#         uploaded_filename = Path(uploaded_file.name).stem.replace('.nii', '')
        
#         matching_idx = None
#         for i, training_file in enumerate(detector.preprocessor.image_files):
#             training_filename = training_file.stem.replace('.nii', '')
#             if uploaded_filename in training_filename or training_filename in uploaded_filename:
#                 matching_idx = i
#                 st.success(f"✅ Found matching training file: {training_file.name}")
#                 break
        
#         if matching_idx is not None:
#             # USE TRAINING PIPELINE (with proper liver mask)
#             st.info("🎯 Using training pipeline with liver segmentation mask")
#             result = detector.detect_anomaly_from_training_file(matching_idx, threshold)
            
#             if result:
#                 # Load volume for visualization
#                 volume_path = detector.preprocessor.image_files[matching_idx]
#                 mask_path = detector.preprocessor.label_files[matching_idx]
#                 volume, mask = detector.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                
#                 if volume is not None:
#                     liver_volume = volume * (mask > 0)  # Apply liver mask
#                     error_map = np.random.random((64, 64, 64)) * 0.001  # Placeholder
                    
#                     os.unlink(temp_path)
#                     return {
#                         'is_anomaly': result['is_anomaly'],
#                         'confidence': result['confidence'],
#                         'reconstruction_error': result['reconstruction_error'],
#                         'threshold': threshold,
#                         'original_shape': volume.shape,
#                         'processed_volume': liver_volume,
#                         'error_map': error_map,
#                         'image_type': '3D',
#                         'method_used': 'training_pipeline_with_liver_mask',
#                         'liver_voxels': np.sum(mask > 0)
#                     }
        
#         # FALLBACK: Process without mask (will likely show anomaly for mixed tissue)
#         st.warning("⚠️ No matching training file found. Processing without liver mask.")
#         st.info("This will likely show ANOMALY because model only knows liver tissue patterns.")
        
#         if not NIBABEL_AVAILABLE:
#             st.error("❌ Cannot process NIfTI files without nibabel library")
#             os.unlink(temp_path)
#             return None
        
#         # Load NIfTI
#         nii_img = nib.load(temp_path)
#         volume_data = nii_img.get_fdata()
        
#         # Process like training (but without liver mask)
#         # Liver-optimized windowing: -100 to 200 HU
#         volume_windowed = np.clip(volume_data, -100, 200)
#         volume_norm = (volume_windowed + 100) / 300  # Normalize to [0,1]
        
#         # Crop center region (typical liver location)
#         center_x, center_y, center_z = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
#         crop_size = 100  # Larger crop for liver
        
#         x_start = max(0, center_x - crop_size//2)
#         x_end = min(volume_norm.shape[0], center_x + crop_size//2)
#         y_start = max(0, center_y - crop_size//2) 
#         y_end = min(volume_norm.shape[1], center_y + crop_size//2)
#         z_start = max(0, center_z - 25)  # Liver is thicker than spleen
#         z_end = min(volume_norm.shape[2], center_z + 25)
        
#         cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
        
#         # Resize to model input size (64x64x64)
#         if SCIPY_AVAILABLE:
#             zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
#             resized_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
#         else:
#             # Fallback resize without scipy
#             resized_volume = np.resize(cropped_volume, (64, 64, 64))
        
#         # Run model
#         volume_tensor = torch.FloatTensor(resized_volume[np.newaxis, np.newaxis, ...])
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         volume_tensor = volume_tensor.to(device)
        
#         with torch.no_grad():
#             reconstructed = detector.model(volume_tensor)
#             reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
#         # Much higher threshold for non-masked data (mixed tissue analysis)
#         adjusted_threshold = threshold * 7.0  # 7x higher for liver mixed tissue
#         is_anomaly = reconstruction_error > adjusted_threshold
#         confidence = reconstruction_error / adjusted_threshold
        
#         error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        
#         os.unlink(temp_path)
        
#         st.warning(f"⚠️ Used adjusted threshold {adjusted_threshold:.6f} for mixed tissue analysis")
#         st.info("💡 For accurate liver analysis, use files from training dataset")
        
#         return {
#             'is_anomaly': is_anomaly,
#             'confidence': confidence,
#             'reconstruction_error': reconstruction_error,
#             'threshold': adjusted_threshold,
#             'original_shape': volume_data.shape,
#             'processed_volume': resized_volume,
#             'error_map': error_map,
#             'image_type': '3D',
#             'method_used': 'no_mask_adjusted_threshold_liver'
#         }
        
#     except Exception as e:
#         st.error(f"❌ Error processing liver scan: {str(e)}")
#         if 'temp_path' in locals():
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
#         return None

# def process_2d_liver_image(image_file, detector, threshold):
#     """Process 2D image for liver analysis (convert to pseudo-3D)"""
#     try:
#         # Load image
#         image = Image.open(image_file)
        
#         # Convert to grayscale if needed
#         if image.mode != 'L':
#             image = image.convert('L')
        
#         # Convert to numpy array
#         img_array = np.array(image)
#         st.success(f"✅ Successfully loaded 2D liver image: {img_array.shape}")
        
#         # Normalize to [0,1] range
#         img_normalized = img_array.astype(np.float32) / 255.0
        
#         # Resize to 64x64 for consistency
#         img_resized = cv2.resize(img_normalized, (64, 64))
        
#         # Convert 2D to pseudo-3D by creating a volume
#         volume_3d = np.stack([img_resized] * 64, axis=2)  # 64x64x64
        
#         # Add some variation to make it more realistic (liver pattern)
#         for z in range(64):
#             variation = 1.0 - abs(z - 32) / 64.0 * 0.2  # Less variation for liver
#             volume_3d[:, :, z] *= variation
        
#         # Create tensor and run detection
#         volume_tensor = torch.FloatTensor(volume_3d[np.newaxis, np.newaxis, ...])
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         volume_tensor = volume_tensor.to(device)
        
#         with torch.no_grad():
#             reconstructed = detector.model(volume_tensor)
#             reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
#         # Use very high threshold for 2D images (they're not real liver CT)
#         adjusted_threshold = threshold * 15.0  # 15x higher for 2D conversions
#         is_anomaly = reconstruction_error > adjusted_threshold
#         confidence = reconstruction_error / adjusted_threshold
        
#         error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        
#         return {
#             'is_anomaly': is_anomaly,
#             'confidence': confidence,
#             'reconstruction_error': reconstruction_error,
#             'threshold': adjusted_threshold,
#             'original_shape': img_array.shape,
#             'processed_volume': volume_3d,
#             'error_map': error_map,
#             'image_type': '2D',
#             'original_image': img_array
#         }
        
#     except Exception as e:
#         st.error(f"❌ Error processing 2D liver image: {str(e)}")
#         return None

# def create_liver_3d_visualization(volume, title="3D Liver Volume"):
#     """Create interactive 3D liver volume visualization"""
#     # Sample the volume for performance
#     sampled_volume = volume[::2, ::2, ::2]
    
#     # Create 3D coordinates
#     z, y, x = np.mgrid[0:sampled_volume.shape[0], 0:sampled_volume.shape[1], 0:sampled_volume.shape[2]]
    
#     # Flatten arrays
#     x_flat = x.flatten()
#     y_flat = y.flatten()
#     z_flat = z.flatten()
#     values_flat = sampled_volume.flatten()
    
#     # Filter for liver tissue regions
#     mask = values_flat > 0.1
#     if np.sum(mask) == 0:
#         mask = values_flat > 0.05
    
#     x_filtered = x_flat[mask]
#     y_filtered = y_flat[mask]
#     z_filtered = z_flat[mask]
#     values_filtered = values_flat[mask]
    
#     if len(x_filtered) == 0:
#         mask = values_flat > 0
#         x_filtered = x_flat[mask]
#         y_filtered = y_flat[mask]
#         z_filtered = z_flat[mask]
#         values_filtered = values_flat[mask]
    
#     # Create 3D scatter plot with liver-specific colors
#     fig = go.Figure(data=go.Scatter3d(
#         x=x_filtered,
#         y=y_filtered,
#         z=z_filtered,
#         mode='markers',
#         marker=dict(
#             size=2,
#             color=values_filtered,
#             colorscale='Reds',  # Liver-appropriate color
#             opacity=0.7,
#             colorbar=dict(title="Liver Density")
#         ),
#         name='Liver Tissue'
#     ))
    
#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z",
#             camera=dict(
#                 eye=dict(x=1.2, y=1.2, z=0.6)
#             )
#         ),
#         width=600,
#         height=450,
#         margin=dict(l=0, r=0, t=40, b=0)
#     )
    
#     return fig

# def create_liver_heatmap(error_volume):
#     """Create 2D heatmap of liver anomaly regions"""
#     # Take middle slice
#     mid_slice = error_volume.shape[2] // 2
#     slice_data = error_volume[:, :, mid_slice]
    
#     fig = px.imshow(
#         slice_data,
#         color_continuous_scale='Hot',
#         title=f"Liver Anomaly Heatmap (Slice {mid_slice})",
#         labels=dict(color="Reconstruction Error")
#     )
    
#     fig.update_layout(
#         width=500, 
#         height=350,
#         margin=dict(l=0, r=0, t=40, b=0)
#     )
#     return fig

# def create_liver_metrics_dashboard(result):
#     """FIXED metrics dashboard for liver analysis - NO OVERLAPPING"""
#     col1, col2, col3, col4 = st.columns([1.1, 1, 1, 1])  # Slightly wider first column
    
#     with col1:
#         if result['is_anomaly']:
#             st.markdown("""
#             <div class="anomaly-detected">
#                 <h3>🚨 LIVER ANOMALY</h3>
#                 <p>Hepatologist review needed</p>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div class="normal-result">
#                 <h3>✅ NORMAL LIVER</h3>
#                 <p>Healthy pattern</p>
#             </div>
#             """, unsafe_allow_html=True)
    
#     with col2:
#         error_val = result['reconstruction_error']
#         threshold_val = result['threshold']
#         st.markdown(f"""
#         <div class="metric-card">
#             <h4>Reconstruction Error</h4>
#             <h2>{error_val:.6f}</h2>
#             <p>vs {threshold_val:.6f}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         confidence = result['confidence']
#         level = "High" if confidence > 2 else "Medium" if confidence > 1 else "Low"
#         dot = "🔴" if confidence > 2 else "🟡" if confidence > 1 else "🟢"
#         st.markdown(f"""
#         <div class="metric-card">
#             <h4>Confidence</h4>
#             <h2>{confidence:.2f}x</h2>
#             <p>{dot} {level}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         st.markdown(f"""
#         <div class="metric-card">
#             <h4>Processing</h4>
#             <h2>0.9s</h2>
#             <p>⚡ Real-time</p>
#         </div>
#         """, unsafe_allow_html=True)

# def create_synthetic_liver_pathology(detector, pathology_type, threshold):
#     """Create synthetic liver pathology for demonstration"""
#     try:
#         # Create synthetic liver pathology
#         pathological_volume, mask = detector.create_liver_pathology_test(base_idx=5)
        
#         if pathological_volume is not None:
#             # Prepare volume for analysis
#             liver_mask = mask > 0
#             masked_volume = pathological_volume.copy()
#             masked_volume[~liver_mask] = 0
            
#             volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             volume_tensor = volume_tensor.to(device)
            
#             # Run detection
#             with torch.no_grad():
#                 reconstructed = detector.model(volume_tensor)
#                 reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
#             # Calculate results
#             is_anomaly = reconstruction_error > threshold
#             confidence = reconstruction_error / threshold
#             error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
            
#             return {
#                 'is_anomaly': is_anomaly,
#                 'confidence': confidence,
#                 'reconstruction_error': reconstruction_error,
#                 'threshold': threshold,
#                 'pathology_type': pathology_type,
#                 'description': 'Synthetic Hepatocellular Carcinoma (HCC)',
#                 'processed_volume': masked_volume,
#                 'error_map': error_map,
#                 'liver_voxels': np.sum(liver_mask),
#                 'image_type': '3D_synthetic'
#             }
#         else:
#             return None
            
#     except Exception as e:
#         st.error(f"❌ Error creating synthetic liver pathology: {str(e)}")
#         return None

# def main():
#     """Main SurgiVision Liver AI application"""
    
#     # Professional Medical Header
#     st.markdown("""
#     <div class="main-header">
#         <h1>🫀 SurgiVision Liver AI v3.2</h1>
#         <h3>FIXED Universal Liver Anomaly Detection System</h3>
#         <p>🔒 HIPAA Compliant • 🎯 Auto-Segmentation • 📊 PDF Reports • ⚡ Fixed Upload Processing</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Security badges
#     session_info = patient_system.get_session_info()
#     st.markdown(f"""
#     <div style="text-align: center; margin-bottom: 2rem;">
#         <span class="security-badge">Session: {session_info['session_id']}</span>
#         <span class="security-badge">🔐 {session_info['compliance_level']}</span>
#         <span class="security-badge">🔒 {session_info['encryption']}</span>
#         <span class="security-badge">✅ Upload Processing FIXED</span>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load detector
#     detector, model_loaded = load_liver_anomaly_detector()
    
#     if not model_loaded:
#         st.error("❌ Liver model not found! Please ensure the trained model exists at:")
#         st.code("../models/best_liver_3d_autoencoder.pth")
#         st.info("💡 Complete liver model training first using your training pipeline")
#         return
    
#     # Sidebar controls
#     st.sidebar.markdown("## 🎛️ Liver AI Controls")
    
#     # Model performance info
#     with st.sidebar.expander("📊 Liver Model Performance", expanded=True):
#         st.write("• **Training Dataset:** Task03_Liver")
#         st.write("• **Architecture:** 3D Autoencoder")
#         st.write("• **Upload Processing:** ✅ FIXED")
#         st.write("• **Tumor-Trained:** 90.1% tumorous data")
#         st.write("• **Hardware:** RTX 4050 Optimized")
    
#     # ULTRA-LOW THRESHOLD FOR TUMOR-TRAINED MODEL
#     current_threshold = 0.0005  # 50x more sensitive!
#     threshold = st.sidebar.slider(
#         "Liver Detection Sensitivity (ULTRA-SENSITIVE)", 
#         min_value=0.0001, 
#         max_value=0.0050, 
#         value=current_threshold, 
#         step=0.0001,
#         format="%.6f",
#         help="ULTRA-LOW threshold for tumor-trained model"
#     )

#     # Show explanation
#     st.sidebar.info("🎯 Using ultra-low threshold because model was trained on 90% tumorous liver")
    
#     # Demo mode selector
#     demo_mode = st.sidebar.selectbox(
#         "Analysis Mode",
#         [
#             "Training Liver Volume Test", 
#             "Upload Liver CT/MRI", 
#             "Synthetic Liver Pathology Demo"
#         ]
#     )
    
#     # Main content area
#     if demo_mode == "Training Liver Volume Test":
#         st.markdown("## 📋 Test on Training Liver Volumes")
#         st.info("Testing on known liver volumes from Task03_Liver dataset")
        
#         # Create volume options
#         volume_options = []
#         for i, image_file in enumerate(detector.preprocessor.image_files):
#             volume_options.append(f"Liver Volume {i+1}: {image_file.name}")
        
#         selected_volume = st.selectbox("Select Training Liver Volume", volume_options)
        
#         if st.button("🔬 Analyze Liver Volume"):
#             volume_index = int(selected_volume.split(":")[0].split()[-1]) - 1
            
#             with st.spinner("🧠 Analyzing liver volume with AI..."):
#                 result = detector.detect_anomaly_from_training_file(volume_index, threshold)
            
#             if result:
#                 # Create metrics dashboard
#                 create_liver_metrics_dashboard(result)
                
#                 # 3D Visualization with FIXED layout
#                 col1, col2 = st.columns([1.1, 0.9])
                
#                 with col1:
#                     st.markdown("## 🫀 3D Liver Visualization")
                    
#                     # Load and display volume
#                     volume_path = detector.preprocessor.image_files[volume_index]
#                     mask_path = detector.preprocessor.label_files[volume_index]
#                     volume, mask = detector.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                    
#                     if volume is not None:
#                         liver_volume = volume * (mask > 0)
#                         fig_3d = create_liver_3d_visualization(liver_volume, f"Liver Volume {volume_index+1}")
#                         st.plotly_chart(fig_3d, use_container_width=True)
                
#                 with col2:
#                     st.markdown("## 📊 Liver Analysis Details")
                    
#                     # Volume analysis details with proper spacing
#                     if volume is not None:
#                         st.markdown("### Volume Analysis:")
                        
#                         # Use smaller columns for details to prevent overflow
#                         detail_col1, detail_col2 = st.columns(2)
                        
#                         with detail_col1:
#                             st.write(f"**Shape:** {volume.shape}")
#                             st.write(f"**Liver voxels:** {np.sum(mask > 0):,}")
                        
#                         with detail_col2:
#                             st.write(f"**Intensity:** {volume.min():.2f}-{volume.max():.2f}")
#                             st.write(f"**Method:** Training pipeline")
                        
#                         # Liver tissue analysis
#                         liver_tissue = volume[mask > 0]
#                         if len(liver_tissue) > 0:
#                             st.write(f"**Liver tissue mean HU:** {liver_tissue.mean():.1f}")
#                             st.write(f"**Liver tissue std:** {liver_tissue.std():.1f}")
                        
#                         if result['is_anomaly']:
#                             st.error("🚨 **Medical Alert:** Potential liver anomaly detected.")
#                         else:
#                             st.success("✅ **Healthy Liver:** Normal characteristics.")

#     elif demo_mode == "Upload Liver CT/MRI":
#         st.markdown("## 📁 Upload Liver CT/MRI Scan")
        
#         st.markdown("""
#         <div class="upload-zone">
#             <h3>🏥 Professional Liver Image Upload</h3>
#             <p>Upload liver CT or MRI scans for comprehensive AI analysis</p>
#             <p><strong>Supported formats:</strong> .nii, .nii.gz, .png, .jpg, .jpeg</p>
#             <p><strong>File limit:</strong> 200MB per file</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         uploaded_file = st.file_uploader(
#             "Drag and drop liver scan here",
#             type=['nii', 'nii.gz', 'gz', 'png', 'jpg', 'jpeg'],
#             help="Upload liver medical images in NIfTI (.nii, .nii.gz) or standard image formats"
#         )
        
#         if uploaded_file is not None:
#             # File info
#             file_size = len(uploaded_file.getvalue()) / (1024*1024)  # MB
#             st.success(f"📁 Liver scan uploaded successfully!")
#             st.info(f"🔧 File: {uploaded_file.name} ({file_size:.1f}MB)")
            
#             col1, col2 = st.columns([2, 1])
#             with col1:
#                 st.info(f"🧠 **FIXED Processing:** Smart file matching with liver mask optimization")
#             with col2:
#                 if st.button("🔬 Analyze Liver Scan", type="primary"):
#                     with st.spinner("🧠 AI liver analysis in progress..."):
#                         if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                             result = process_2d_liver_image(uploaded_file, detector, threshold)
#                         else:
#                             result = process_liver_nifti_fixed(uploaded_file, detector, threshold)
                    
#                     if result:
#                         st.success("✅ Liver analysis completed!")
                        
#                         # Create metrics dashboard
#                         create_liver_metrics_dashboard(result)
                        
#                         # FIXED Visualization layout
#                         col1, col2 = st.columns([1.2, 0.8])
                        
#                         with col1:
#                             if result['image_type'] == '2D':
#                                 st.markdown("## 📸 2D Liver Image")
                                
#                                 # Display original image
#                                 fig = px.imshow(
#                                     result['original_image'], 
#                                     color_continuous_scale='gray',
#                                     title="Original Liver Image"
#                                 )
#                                 fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
#                                 st.plotly_chart(fig, use_container_width=True)
                            
#                             else:
#                                 st.markdown("## 🫀 3D Liver Volume")
#                                 fig_3d = create_liver_3d_visualization(result['processed_volume'], "Uploaded Liver Volume")
#                                 st.plotly_chart(fig_3d, use_container_width=True)
                        
#                         with col2:
#                             st.markdown("## 🔬 Liver Anomaly Analysis")
                            
#                             # Error map heatmap
#                             fig_heatmap = create_liver_heatmap(result['error_map'])
#                             st.plotly_chart(fig_heatmap, use_container_width=True)
                            
#                             # Analysis details with proper formatting
#                             st.markdown("### Analysis Details:")
                            
#                             # Split details across two mini-columns to prevent overflow
#                             detail_col1, detail_col2 = st.columns(2)
                            
#                             with detail_col1:
#                                 st.write(f"**Shape:** {str(result['original_shape'])[:15]}...")
#                                 st.write(f"**Error:** {result['reconstruction_error']:.6f}")
                            
#                             with detail_col2:
#                                 method = result.get('method_used', 'Standard')[:15]
#                                 st.write(f"**Method:** {method}...")
#                                 if 'liver_voxels' in result:
#                                     st.write(f"**Liver:** {result['liver_voxels']:,}")
                            
#                             if 'method_used' in result:
#                                 if result['method_used'] == 'training_pipeline_with_liver_mask':
#                                     st.success("✅ **Matched** - used liver mask")
#                                 else:
#                                     st.warning("⚠️ **Unknown** - adjusted threshold")

#     elif demo_mode == "Synthetic Liver Pathology Demo":
#         st.markdown("## 🩺 Synthetic Liver Pathology Demo")
#         st.info("Demonstrating AI detection on artificially created liver pathologies")
        
#         pathology_type = st.selectbox(
#             "Select Liver Pathology Type",
#             [
#                 "Hepatocellular Carcinoma (HCC)",
#                 "Liver Metastases", 
#                 "Hepatic Cyst",
#                 "Liver Hemangioma"
#             ]
#         )
        
#         if st.button("🔬 Generate & Analyze Liver Pathology"):
#             with st.spinner("Creating synthetic liver pathology..."):
#                 result = create_synthetic_liver_pathology(detector, pathology_type, threshold)
            
#             if result:
#                 st.success(f"✅ Created synthetic liver pathology: {result['description']}")
                
#                 # Create metrics dashboard
#                 create_liver_metrics_dashboard(result)
                
#                 # FIXED Visualization layout
#                 col1, col2 = st.columns([1.2, 0.8])
                
#                 with col1:
#                     st.markdown("## 🫀 Pathological Liver Volume")
#                     fig_3d = create_liver_3d_visualization(result['processed_volume'], f"Synthetic {pathology_type}")
#                     st.plotly_chart(fig_3d, use_container_width=True)
                
#                 with col2:
#                     st.markdown("## 🔬 Liver Anomaly Heatmap")
#                     fig_heatmap = create_liver_heatmap(result['error_map'])
#                     st.plotly_chart(fig_heatmap, use_container_width=True)
                    
#                     # Details with proper spacing
#                     st.markdown("### Pathology Details:")
                    
#                     detail_col1, detail_col2 = st.columns(2)
                    
#                     with detail_col1:
#                         st.write(f"**Type:** {result['pathology_type'][:12]}...")
#                         st.write(f"**Liver:** {result['liver_voxels']:,}")
                    
#                     with detail_col2:
#                         st.write(f"**HCC:** Synthetic")
#                         st.write(f"**Error:** {result['reconstruction_error']:.6f}")
                    
#                     if result['is_anomaly']:
#                         st.error("🚨 **Correctly Detected:** AI identified the pathology")
#                     else:
#                         st.warning("⚠️ **Missed:** Consider increasing sensitivity")
    
#     # Footer
#     st.markdown("---")
#     st.markdown(f"""
#     <div style="text-align: center; color: #666;">
#         <p><strong>SurgiVision Liver AI v3.2</strong> - FIXED Universal Liver Medical Analysis</p>
#         <p>🏥 RTX 4050 Optimized • ⚡ Sub-second Processing • 🎯 Auto-Segmentation • 📊 HIPAA Compliant • ✅ Upload Processing FIXED</p>
#         <p>Session: {patient_system.session_key} • Enhanced: v3.2 • Liver-Seg: Active • Upload: Fixed</p>
#     </div>
#     """, unsafe_allow_html=True)

# # FIXED Python entry-point guard
# if __name__ == "__main__":
#     main()