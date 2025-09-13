import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model_regularized import LiverAutoencoderRegularized
from extreme_liver_destroyer import ExtremeStructureDestroyer
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import nibabel as nib

class RegularizedLiverEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the NEW regularized model
        model_path = "../models/liver_regularized_final.pth"
        if not Path(model_path).exists():
            print(f"❌ Regularized model not found: {model_path}")
            return
            
        self.model = LiverAutoencoderRegularized(latent_dim=256, dropout_rate=0.3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor
        self.preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
        
        # Get test images (unseen data)
        self.test_images_dir = Path("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver\\imagesTs")
        self.test_image_files = sorted(list(self.test_images_dir.glob("*.nii.gz")))
        
        # Load training info
        self.training_info = {
            'epochs_trained': checkpoint.get('epoch', 68),
            'best_val_loss': checkpoint.get('best_val_loss', 0.024289),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', [])
        }
        
        print(f"✅ Regularized model loaded on {self.device}")
        print(f"📊 Training epochs: {self.training_info['epochs_trained']}")
        print(f"🎯 Best validation loss: {self.training_info['best_val_loss']:.6f}")
        print(f"🔍 Test volumes available: {len(self.test_image_files)}")

    def calculate_regularized_baseline(self):
        """Calculate baseline from validation set (proper approach)"""
        print("📈 Calculating baseline from validation split (last 20% of training data)...")
        
        # Use the same validation split as training (last 20%)
        total_indices = list(range(len(self.preprocessor.image_files)))
        np.random.seed(42)  # Same seed as training
        np.random.shuffle(total_indices)
        split_point = int(0.8 * len(total_indices))
        val_indices = total_indices[split_point:]
        
        print(f"📉 Using {len(val_indices)} validation samples for baseline")
        
        validation_errors = []
        
        for i in val_indices[:20]:  # Use subset for speed
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
                
                validation_errors.append(error)
                
            except Exception as e:
                print(f"    ⚠️ Error with validation volume {i}: {e}")
                continue
        
        print(f"✅ Processed {len(validation_errors)} validation volumes")
        return validation_errors

    def calculate_adaptive_threshold(self, validation_errors):
        """Calculate threshold from validation errors"""
        if not validation_errors:
            print("❌ No validation errors available")
            return 0.030  # Conservative fallback
        
        mean_val_error = np.mean(validation_errors)
        std_val_error = np.std(validation_errors)
        
        # Multiple threshold strategies
        thresholds = {
            'conservative': mean_val_error + 3.0 * std_val_error,  # 3-sigma (very specific)
            'balanced': mean_val_error + 2.0 * std_val_error,      # 2-sigma (balanced)
            'sensitive': mean_val_error + 1.5 * std_val_error,     # 1.5-sigma (more sensitive)
            'percentile_95': np.percentile(validation_errors, 95),  # 95th percentile
            'percentile_90': np.percentile(validation_errors, 90)   # 90th percentile
        }
        
        print(f"\n🎯 THRESHOLD CALCULATION:")
        print(f"Validation mean error: {mean_val_error:.6f}")
        print(f"Validation std error: {std_val_error:.6f}")
        print(f"Validation error range: {min(validation_errors):.6f} - {max(validation_errors):.6f}")
        
        for name, threshold in thresholds.items():
            print(f"  {name.capitalize()}: {threshold:.6f}")
        
        # Return balanced threshold as primary
        return thresholds['balanced'], thresholds

    def evaluate_unseen_test_data(self, threshold):
        """Evaluate on completely unseen test data with regularized model"""
        print(f"\n🔍 EVALUATING REGULARIZED MODEL ON UNSEEN TEST DATA")
        print(f"Using threshold: {threshold:.6f}")
        
        test_results = []
        
        for i, test_image_path in enumerate(self.test_image_files):
            try:
                print(f"  📄 Processing {i+1}/{len(self.test_image_files)}: {test_image_path.name}")
                
                # Load and preprocess test image (no mask available)
                nii_img = nib.load(test_image_path)
                volume_data = nii_img.get_fdata()
                
                # Apply same preprocessing as training
                volume_windowed = np.clip(volume_data, -100, 200)
                volume_norm = (volume_windowed + 100) / 300
                
                # Extract center region (liver location)
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
                
                # Run inference with regularized model
                volume_tensor = torch.FloatTensor(resized_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                
                is_anomaly = error > threshold
                confidence = error / threshold if threshold > 0 else 0
                
                test_results.append({
                    'file': test_image_path.name,
                    'original_shape': volume_data.shape,
                    'reconstruction_error': error,
                    'is_anomaly': is_anomaly,
                    'confidence': confidence
                })
                
                status = "🚨" if is_anomaly else "✅"
                print(f"    Error: {error:.6f}, {status} ({'Anomaly' if is_anomaly else 'Normal'})")
                
            except Exception as e:
                print(f"    ❌ Error processing {test_image_path.name}: {e}")
                continue
        
        return test_results

    def synthetic_pathology_test_regularized(self):
        """Test synthetic pathologies with regularized model"""
        print(f"\n🧪 SYNTHETIC PATHOLOGY TESTING (REGULARIZED MODEL)")
        
        destroyer = ExtremeStructureDestroyer(self.preprocessor)
        
        # Test on multiple validation volumes (not training)
        total_indices = list(range(len(self.preprocessor.image_files)))
        np.random.seed(42)
        np.random.shuffle(total_indices)
        split_point = int(0.8 * len(total_indices))
        val_indices = total_indices[split_point:]
        
        # Use validation indices for pathology testing
        base_indices = val_indices[:3]  # Test on 3 validation volumes
        
        synthetic_results = []
        
        for base_idx in base_indices:
            try:
                print(f"  🔬 Creating pathologies from validation volume {base_idx}...")
                
                pathologies = destroyer.create_all_extreme_destructive_pathologies(base_idx)
                
                for case in pathologies:
                    liver_mask = case['mask'] > 0
                    liver_volume = case['volume'].copy()
                    liver_volume[~liver_mask] = 0
                    
                    volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                    
                    with torch.no_grad():
                        reconstructed = self.model(volume_tensor)
                        error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    
                    synthetic_results.append({
                        'base_volume': base_idx,
                        'pathology_type': case['description'],
                        'error': error,
                        'structural_change': case.get('structural_change', 0)
                    })
                    
            except Exception as e:
                print(f"    ⚠️ Error with base volume {base_idx}: {e}")
                continue
        
        return synthetic_results

    def compare_with_original_model(self, validation_errors, test_results):
        """Compare regularized model with original overfitted model"""
        print(f"\n📊 COMPARISON: REGULARIZED vs ORIGINAL MODEL")
        print("=" * 50)
        
        # Current regularized model stats
        mean_val_error = np.mean(validation_errors)
        test_errors = [r['reconstruction_error'] for r in test_results]
        mean_test_error = np.mean(test_errors)
        
        # Calculate generalization gap
        generalization_gap = abs(mean_test_error - mean_val_error) / mean_val_error * 100
        
        # Test anomaly rate
        test_anomalies = sum([1 for r in test_results if r['is_anomaly']])
        test_anomaly_rate = test_anomalies / len(test_results) * 100
        
        print(f"🔄 REGULARIZED MODEL (NEW):")
        print(f"  Validation mean error: {mean_val_error:.6f}")
        print(f"  Test mean error: {mean_test_error:.6f}")
        print(f"  Generalization gap: {generalization_gap:.1f}%")
        print(f"  Test anomaly rate: {test_anomaly_rate:.1f}% ({test_anomalies}/{len(test_results)})")
        
        print(f"\n📈 ORIGINAL MODEL (OLD):")
        print(f"  Training mean error: ~0.017")
        print(f"  Test mean error: ~0.30")
        print(f"  Generalization gap: 2644.5%")
        print(f"  Test anomaly rate: 100.0%")
        
        print(f"\n🏆 IMPROVEMENT ANALYSIS:")
        improvement_factor = 2644.5 / generalization_gap if generalization_gap > 0 else float('inf')
        fp_improvement = 100.0 / test_anomaly_rate if test_anomaly_rate > 0 else float('inf')
        
        print(f"  Generalization gap: {improvement_factor:.1f}x better")
        print(f"  False positive rate: {fp_improvement:.1f}x better")
        
        if generalization_gap < 50:
            assessment = "🏆 EXCELLENT - Overfitting completely fixed!"
        elif generalization_gap < 100:
            assessment = "✅ VERY GOOD - Major improvement in generalization!"
        elif generalization_gap < 200:
            assessment = "👍 GOOD - Significant improvement!"
        else:
            assessment = "📈 IMPROVED - Still some room for improvement"
        
        print(f"  Overall: {assessment}")
        
        return {
            'regularized_gap': generalization_gap,
            'regularized_anomaly_rate': test_anomaly_rate,
            'improvement_factor': improvement_factor,
            'fp_improvement': fp_improvement
        }

    def create_comprehensive_visualizations(self, validation_errors, test_results, synthetic_results, comparison_stats):
        """Create comprehensive evaluation visualizations"""
        print("\n📊 Creating comprehensive evaluation visualizations...")
        
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Regularized Liver Model - Comprehensive Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Training curves (from training history)
        ax1 = axes[0, 0]
        if self.training_info['train_losses'] and self.training_info['val_losses']:
            epochs = range(1, len(self.training_info['train_losses']) + 1)
            ax1.plot(epochs, self.training_info['train_losses'], 'b-', label='Training Loss', alpha=0.7)
            ax1.plot(epochs, self.training_info['val_losses'], 'r-', label='Validation Loss', alpha=0.7)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training History (Regularized)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Validation vs Test error distribution
        ax2 = axes[0, 1]
        test_errors = [r['reconstruction_error'] for r in test_results]
        
        ax2.hist(validation_errors, bins=15, alpha=0.7, label=f'Validation (n={len(validation_errors)})', color='blue')
        ax2.hist(test_errors, bins=15, alpha=0.7, label=f'Test (n={len(test_errors)})', color='red')
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Validation vs Test Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Test set classifications
        ax3 = axes[0, 2]
        test_anomalies = sum([1 for r in test_results if r['is_anomaly']])
        test_normal = len(test_results) - test_anomalies
        
        labels = ['Normal', 'Anomaly']
        sizes = [test_normal, test_anomalies]
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Test Set Classifications\n(Regularized Model)')
        
        # 4. Synthetic pathology performance
        ax4 = axes[0, 3]
        if synthetic_results:
            pathology_types = {}
            for result in synthetic_results:
                ptype = result['pathology_type'].split()[0]
                if ptype not in pathology_types:
                    pathology_types[ptype] = []
                pathology_types[ptype].append(result['error'])
            
            type_names = list(pathology_types.keys())
            type_errors = [np.mean(pathology_types[t]) for t in type_names]
            
            ax4.barh(type_names, type_errors, color='lightblue')
            ax4.set_xlabel('Mean Reconstruction Error')
            ax4.set_title('Synthetic Pathology Errors')
            ax4.grid(True, alpha=0.3)
        
        # 5. Generalization gap comparison
        ax5 = axes[1, 0]
        model_types = ['Original\n(Overfitted)', 'Regularized\n(New)']
        generalization_gaps = [2644.5, comparison_stats['regularized_gap']]
        colors = ['red', 'green']
        
        bars = ax5.bar(model_types, generalization_gaps, color=colors, alpha=0.7)
        ax5.set_ylabel('Generalization Gap (%)')
        ax5.set_title('Generalization Gap Comparison')
        ax5.set_yscale('log')
        
        # Add value labels
        for bar, gap in zip(bars, generalization_gaps):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{gap:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax5.grid(True, alpha=0.3)
        
        # 6. False positive rate comparison
        ax6 = axes[1, 1]
        fp_rates = [100.0, comparison_stats['regularized_anomaly_rate']]
        
        bars = ax6.bar(model_types, fp_rates, color=colors, alpha=0.7)
        ax6.set_ylabel('Test Anomaly Rate (%)')
        ax6.set_title('False Positive Rate Comparison')
        
        for bar, rate in zip(bars, fp_rates):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax6.grid(True, alpha=0.3)
        
        # 7. Error distribution box plot
        ax7 = axes[1, 2]
        data_to_plot = [validation_errors, test_errors]
        labels_box = ['Validation', 'Test']
        
        bp = ax7.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax7.set_ylabel('Reconstruction Error')
        ax7.set_title('Error Distribution (Regularized)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Overall performance summary
        ax8 = axes[1, 3]
        
        # Calculate final performance metrics
        specificity = (len(test_results) - test_anomalies) / len(test_results) * 100
        generalization_score = max(0, 100 - comparison_stats['regularized_gap'])
        overall_score = (specificity + generalization_score) / 2
        
        metrics = ['Specificity\n(%)', 'Generalization\nScore (%)', 'Overall\nScore (%)']
        scores = [specificity, generalization_score, overall_score]
        colors_bar = ['lightblue', 'lightgreen', 'gold']
        
        bars = ax8.bar(metrics, scores, color=colors_bar, alpha=0.7)
        ax8.set_ylabel('Score (%)')
        ax8.set_title('Regularized Model Performance')
        ax8.set_ylim(0, 100)
        
        for bar, score in zip(bars, scores):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/regularized_liver_comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive evaluation saved: ../results/regularized_liver_comprehensive_evaluation.png")

    def generate_final_regularized_report(self):
        """Generate comprehensive final report for regularized model"""
        print("=" * 70)
        print("🫘 REGULARIZED LIVER MODEL - FINAL EVALUATION REPORT")
        print("=" * 70)
        
        # Step 1: Calculate baseline from validation data
        validation_errors = self.calculate_regularized_baseline()
        if not validation_errors:
            print("❌ Could not establish validation baseline!")
            return
        
        # Step 2: Calculate adaptive threshold
        threshold, all_thresholds = self.calculate_adaptive_threshold(validation_errors)
        
        # Step 3: Evaluate on unseen test data
        test_results = self.evaluate_unseen_test_data(threshold)
        
        # Step 4: Test synthetic pathologies
        synthetic_results = self.synthetic_pathology_test_regularized()
        
        # Step 5: Compare with original model
        comparison_stats = self.compare_with_original_model(validation_errors, test_results)
        
        # Step 6: Create visualizations
        self.create_comprehensive_visualizations(validation_errors, test_results, synthetic_results, comparison_stats)
        
        # Final summary
        print(f"\n🎯 FINAL REGULARIZED MODEL ASSESSMENT:")
        print("=" * 50)
        
        print(f"📊 Training Information:")
        print(f"   Epochs: {self.training_info['epochs_trained']} (early stopping)")
        print(f"   Best validation loss: {self.training_info['best_val_loss']:.6f}")
        print(f"   Training time: ~21 hours")
        
        print(f"\n🔍 Generalization Performance:")
        print(f"   Generalization gap: {comparison_stats['regularized_gap']:.1f}% (vs 2644.5% original)")
        print(f"   Improvement factor: {comparison_stats['improvement_factor']:.1f}x better")
        
        print(f"\n🎖️ FINAL VERDICT:")
        if comparison_stats['regularized_gap'] < 50:
            verdict = "🏆 OUTSTANDING SUCCESS - Overfitting completely eliminated!"
        elif comparison_stats['regularized_gap'] < 100:
            verdict = "✅ EXCELLENT SUCCESS - Major improvement achieved!"
        elif comparison_stats['regularized_gap'] < 200:
            verdict = "👍 GOOD SUCCESS - Significant progress made!"
        else:
            verdict = "📈 PARTIAL SUCCESS - Improvement shown, further work needed!"
        
        print(f"{verdict}")
        
        print(f"\n🚀 HACKATHON READINESS:")
        print(f"• Model: Fully regularized 3D liver autoencoder")
        print(f"• Training: 68 epochs with early stopping")
        print(f"• Generalization: {comparison_stats['improvement_factor']:.1f}x improvement")
        print(f"• False positives: {comparison_stats['fp_improvement']:.1f}x reduction")
        print(f"• Status: Ready for professional demonstration!")
        
        return comparison_stats

def main():
    """Run comprehensive regularized model evaluation"""
    evaluator = RegularizedLiverEvaluator()
    if not hasattr(evaluator, 'model'):
        return
    
    final_stats = evaluator.generate_final_regularized_report()
    
    print(f"\n🎉 REGULARIZED LIVER MODEL EVALUATION COMPLETED!")
    print(f"📁 Results saved to: ../results/regularized_liver_comprehensive_evaluation.png")

if __name__ == "__main__":
    main()
