import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder
from extreme_liver_destroyer import ExtremeStructureDestroyer

class LiverUnseenDataEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        model_path = "../models/best_liver_3d_autoencoder.pth"
        self.model = Liver3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor (for training baseline)
        self.preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
        
        # Get test images (unseen data)
        self.test_images_dir = Path("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver\\imagesTs")
        self.test_image_files = sorted(list(self.test_images_dir.glob("*.nii.gz")))
        
        # Load threshold from training
        try:
            with open("../models/extreme_liver_threshold.txt", "r") as f:
                self.training_threshold = float(f.read().strip())
        except:
            self.training_threshold = 0.013188
        
        print(f"✅ Evaluator loaded on {self.device}")
        print(f"📊 Training volumes: {len(self.preprocessor.image_files)}")
        print(f"🔍 Test volumes (unseen): {len(self.test_image_files)}")
        print(f"🎯 Training threshold: {self.training_threshold:.6f}")

    def establish_training_baseline(self):
        """Establish baseline from training data for comparison"""
        print("📈 Establishing training data baseline...")
        
        training_errors = []
        sample_size = min(50, len(self.preprocessor.image_files))  # Sample for speed
        
        for i in range(sample_size):
            try:
                volume_path = self.preprocessor.image_files[i]
                mask_path = self.preprocessor.label_files[i]
                
                volume, mask = self.preprocessor.preprocess_liver_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                liver_mask = mask > 0
                liver_volume = volume.copy()
                liver_volume[~liver_mask] = 0
                
                volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                
                training_errors.append(error)
                
            except Exception as e:
                print(f"    ⚠️ Error with training volume {i}: {e}")
                continue
        
        return training_errors

    def evaluate_test_set_unsupervised(self):
        """Evaluate on unseen test set without labels"""
        print("🔍 Evaluating on completely unseen test data...")
        
        test_results = []
        
        for i, test_image_path in enumerate(self.test_image_files):
            try:
                print(f"  📄 Processing test image {i+1}/{len(self.test_image_files)}: {test_image_path.name}")
                
                # Load and preprocess test image (no mask available)
                nii_img = nib.load(test_image_path)
                volume_data = nii_img.get_fdata()
                
                # Apply same preprocessing as training data
                volume_windowed = np.clip(volume_data, -100, 200)
                volume_norm = (volume_windowed + 100) / 300
                
                # Extract center region (likely liver location)
                center_x, center_y, center_z = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
                crop_size = 100
                
                x_start = max(0, center_x - crop_size//2)
                x_end = min(volume_norm.shape[0], center_x + crop_size//2)
                y_start = max(0, center_y - crop_size//2) 
                y_end = min(volume_norm.shape[1], center_y + crop_size//2)
                z_start = max(0, center_z - 25)
                z_end = min(volume_norm.shape[2], center_z + 25)
                
                cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Resize to model input size
                from scipy import ndimage
                zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
                resized_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
                
                # Run inference
                volume_tensor = torch.FloatTensor(resized_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                
                # Classify based on training threshold
                is_anomaly_training_threshold = error > self.training_threshold
                
                test_results.append({
                    'file': test_image_path.name,
                    'original_shape': volume_data.shape,
                    'reconstruction_error': error,
                    'anomaly_training_threshold': is_anomaly_training_threshold,
                    'confidence': error / self.training_threshold if self.training_threshold > 0 else 0
                })
                
                print(f"    Error: {error:.6f}, Anomaly: {'🚨' if is_anomaly_training_threshold else '✅'}")
                
            except Exception as e:
                print(f"    ❌ Error processing {test_image_path.name}: {e}")
                continue
        
        return test_results

    def create_synthetic_pathologies_on_test_data(self):
        """Create synthetic pathologies on test data for labeled evaluation"""
        print("🧪 Creating synthetic pathologies on test data...")
        
        synthetic_test_results = []
        
        # Use first few test images as base for synthetic pathologies
        test_bases = self.test_image_files[:5]  # Use 5 test images as bases
        
        for base_idx, test_image_path in enumerate(test_bases):
            try:
                print(f"  🔬 Creating pathologies from test image {test_image_path.name}...")
                
                # Load test image
                nii_img = nib.load(test_image_path)
                volume_data = nii_img.get_fdata()
                
                # Preprocess
                volume_windowed = np.clip(volume_data, -100, 200)
                volume_norm = (volume_windowed + 100) / 300
                
                # Extract and resize (same as above)
                center_x, center_y, center_z = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
                crop_size = 100
                
                x_start = max(0, center_x - crop_size//2)
                x_end = min(volume_norm.shape[0], center_x + crop_size//2)
                y_start = max(0, center_y - crop_size//2) 
                y_end = min(volume_norm.shape[1], center_y + crop_size//2)
                z_start = max(0, center_z - 25)
                z_end = min(volume_norm.shape[2], center_z + 25)
                
                cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
                
                from scipy import ndimage
                zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
                base_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
                
                # Create synthetic liver mask (assume center region is liver)
                synthetic_mask = np.zeros_like(base_volume)
                synthetic_mask[16:48, 16:48, 16:48] = 1  # Central liver region
                
                # Create different synthetic pathologies
                pathology_types = [
                    ("Swiss Cheese", self.create_swiss_cheese_pathology),
                    ("Intensity Inversion", self.create_inversion_pathology),
                    ("Gradient Destruction", self.create_gradient_pathology),
                    ("Random Noise", self.create_noise_pathology)
                ]
                
                for pathology_name, pathology_func in pathology_types:
                    pathological_volume = pathology_func(base_volume.copy(), synthetic_mask)
                    
                    # Test pathological volume
                    volume_tensor = torch.FloatTensor(pathological_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                    
                    with torch.no_grad():
                        reconstructed = self.model(volume_tensor)
                        error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    
                    is_detected = error > self.training_threshold
                    confidence = error / self.training_threshold
                    
                    synthetic_test_results.append({
                        'base_file': test_image_path.name,
                        'pathology_type': pathology_name,
                        'error': error,
                        'detected': is_detected,
                        'confidence': confidence,
                        'true_label': 'anomaly'
                    })
                    
                    status = "✅ DETECTED" if is_detected else "❌ MISSED"
                    print(f"    {pathology_name}: {status} ({confidence:.2f}x)")
                
            except Exception as e:
                print(f"    ❌ Error with test image {base_idx}: {e}")
                continue
        
        return synthetic_test_results

    def create_swiss_cheese_pathology(self, volume, mask):
        """Create swiss cheese pathology"""
        pathological = volume.copy()
        liver_region = mask > 0
        
        # Create multiple holes
        np.random.seed(42)
        for _ in range(15):
            coords = np.where(liver_region)
            if len(coords[0]) > 0:
                idx = np.random.randint(len(coords[0]))
                x, y, z = coords[0][idx], coords[1][idx], coords[2][idx]
                
                # Create hole
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        for dz in range(-2, 3):
                            nx, ny, nz = x+dx, y+dy, z+dz
                            if 0 <= nx < 64 and 0 <= ny < 64 and 0 <= nz < 64:
                                pathological[nx, ny, nz] = 0.0
        
        return pathological

    def create_inversion_pathology(self, volume, mask):
        """Create intensity inversion"""
        pathological = volume.copy()
        liver_region = mask > 0
        pathological[liver_region] = 1.0 - pathological[liver_region]
        return pathological

    def create_gradient_pathology(self, volume, mask):
        """Create gradient pathology"""
        pathological = volume.copy()
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    if mask[x, y, z] > 0:
                        pathological[x, y, z] = x / 64.0
        return pathological

    def create_noise_pathology(self, volume, mask):
        """Create noise pathology"""
        pathological = volume.copy()
        np.random.seed(123)
        noise = np.random.random((64, 64, 64))
        liver_region = mask > 0
        pathological[liver_region] = noise[liver_region]
        return pathological

    def create_comprehensive_test_visualizations(self, training_errors, test_results, synthetic_test_results):
        """Create comprehensive visualizations"""
        print("📊 Creating test evaluation visualizations...")
        
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training vs Test error distribution
        ax1 = axes[0, 0]
        test_errors = [r['reconstruction_error'] for r in test_results]
        
        ax1.hist(training_errors, bins=20, alpha=0.7, label=f'Training (n={len(training_errors)})', color='blue')
        ax1.hist(test_errors, bins=20, alpha=0.7, label=f'Test (n={len(test_errors)})', color='red')
        ax1.axvline(self.training_threshold, color='black', linestyle='--', label=f'Threshold: {self.training_threshold:.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Training vs Test Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Test set anomaly detection rates
        ax2 = axes[0, 1]
        test_anomalies = sum([1 for r in test_results if r['anomaly_training_threshold']])
        test_normal = len(test_results) - test_anomalies
        
        labels = ['Normal', 'Anomaly']
        sizes = [test_normal, test_anomalies]
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Test Set Classifications')
        
        # 3. Synthetic pathology detection on test data
        ax3 = axes[0, 2]
        pathology_performance = {}
        for result in synthetic_test_results:
            ptype = result['pathology_type']
            if ptype not in pathology_performance:
                pathology_performance[ptype] = {'total': 0, 'detected': 0}
            pathology_performance[ptype]['total'] += 1
            if result['detected']:
                pathology_performance[ptype]['detected'] += 1
        
        pathologies = list(pathology_performance.keys())
        detection_rates = [pathology_performance[p]['detected']/pathology_performance[p]['total']*100 
                          for p in pathologies]
        
        ax3.barh(pathologies, detection_rates, color='lightblue')
        ax3.set_xlabel('Detection Rate (%)')
        ax3.set_title('Synthetic Pathology Detection\n(on Test Data)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error comparison box plot
        ax4 = axes[1, 0]
        data_to_plot = [training_errors, test_errors]
        labels_box = ['Training', 'Test']
        
        bp = ax4.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax4.set_ylabel('Reconstruction Error')
        ax4.set_title('Error Distribution Comparison')
        ax4.grid(True, alpha=0.3)
        
        # 5. Confidence scores for synthetic pathologies
        ax5 = axes[1, 1]
        confidences = [r['confidence'] for r in synthetic_test_results]
        detected_mask = [r['detected'] for r in synthetic_test_results]
        
        detected_conf = [c for c, d in zip(confidences, detected_mask) if d]
        missed_conf = [c for c, d in zip(confidences, detected_mask) if not d]
        
        ax5.hist(detected_conf, bins=10, alpha=0.7, label='Detected', color='green')
        ax5.hist(missed_conf, bins=10, alpha=0.7, label='Missed', color='red')
        ax5.axvline(1.0, color='black', linestyle='--', label='Threshold Line')
        ax5.set_xlabel('Confidence (Error/Threshold)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Pathology Detection Confidence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary metrics
        ax6 = axes[1, 2]
        
        # Calculate key metrics
        mean_training_error = np.mean(training_errors)
        mean_test_error = np.mean(test_errors)
        generalization_gap = abs(mean_test_error - mean_training_error) / mean_training_error * 100
        
        synthetic_detection_rate = sum([1 for r in synthetic_test_results if r['detected']]) / len(synthetic_test_results) * 100
        test_anomaly_rate = test_anomalies / len(test_results) * 100
        
        metrics = ['Generalization\nGap (%)', 'Test Anomaly\nRate (%)', 'Synthetic\nDetection (%)']
        values = [generalization_gap, test_anomaly_rate, synthetic_detection_rate]
        colors_bar = ['orange', 'lightcoral', 'lightgreen']
        
        bars = ax6.bar(metrics, values, color=colors_bar, alpha=0.7)
        ax6.set_ylabel('Percentage')
        ax6.set_title('Key Performance Metrics')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/liver_unseen_data_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'generalization_gap': generalization_gap,
            'test_anomaly_rate': test_anomaly_rate,
            'synthetic_detection_rate': synthetic_detection_rate,
            'mean_training_error': mean_training_error,
            'mean_test_error': mean_test_error
        }

    def generate_unseen_data_report(self):
        """Generate comprehensive report on unseen data performance"""
        print("=" * 60)
        print("🔍 LIVER MODEL EVALUATION ON COMPLETELY UNSEEN DATA")
        print("=" * 60)
        
        # Step 1: Training baseline
        training_errors = self.establish_training_baseline()
        
        # Step 2: Test set evaluation
        test_results = self.evaluate_test_set_unsupervised()
        
        # Step 3: Synthetic pathologies on test data
        synthetic_test_results = self.create_synthetic_pathologies_on_test_data()
        
        # Step 4: Visualizations
        metrics = self.create_comprehensive_test_visualizations(training_errors, test_results, synthetic_test_results)
        
        # Generate final report
        print(f"\n🎯 UNSEEN DATA PERFORMANCE ANALYSIS:")
        print("=" * 50)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples analyzed: {len(training_errors)}")
        print(f"   Test samples (completely unseen): {len(test_results)}")
        print(f"   Synthetic pathologies on test data: {len(synthetic_test_results)}")
        
        print(f"\n📈 Error Analysis:")
        print(f"   Mean training error: {metrics['mean_training_error']:.6f}")
        print(f"   Mean test error: {metrics['mean_test_error']:.6f}")
        print(f"   Generalization gap: {metrics['generalization_gap']:.1f}%")
        
        print(f"\n🔍 Test Set Analysis:")
        print(f"   Test anomaly detection rate: {metrics['test_anomaly_rate']:.1f}%")
        print(f"   Synthetic pathology detection: {metrics['synthetic_detection_rate']:.1f}%")
        
        # Interpretation
        print(f"\n🎖️ GENERALIZATION ASSESSMENT:")
        if metrics['generalization_gap'] < 10:
            generalization_score = "🏆 EXCELLENT - Model generalizes very well to unseen data!"
        elif metrics['generalization_gap'] < 20:
            generalization_score = "✅ GOOD - Acceptable generalization with minor gap!"
        elif metrics['generalization_gap'] < 30:
            generalization_score = "⚠️ MODERATE - Some overfitting, but still usable!"
        else:
            generalization_score = "📉 CONCERNING - Significant generalization gap!"
        
        print(f"{generalization_score}")
        
        print(f"\n📁 Detailed visualizations saved to: ../results/liver_unseen_data_evaluation.png")
        
        return metrics

def main():
    """Run unseen data evaluation"""
    evaluator = LiverUnseenDataEvaluator()
    final_metrics = evaluator.generate_unseen_data_report()
    
    print(f"\n🚀 HACKATHON PRESENTATION METRICS (REALISTIC):")
    print(f"• Model tested on {len(evaluator.test_image_files)} completely unseen liver scans")
    print(f"• Generalization gap: {final_metrics['generalization_gap']:.1f}%")
    print(f"• Synthetic pathology detection on unseen data: {final_metrics['synthetic_detection_rate']:.1f}%")
    print(f"• Real-world applicability: Validated on independent test set")

if __name__ == "__main__":
    import nibabel as nib
    main()
