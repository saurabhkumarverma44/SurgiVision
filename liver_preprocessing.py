import nibabel as nib
import numpy as np
from scipy import ndimage
import os
from pathlib import Path
import matplotlib.pyplot as plt

class LiverDataPreprocessor:
    def __init__(self, data_root):
        """
        Initialize liver data preprocessor
        data_root: path to Task03_Liver folder
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "imagesTr"
        self.labels_dir = self.data_root / "labelsTr" 
        self.test_images_dir = self.data_root / "imagesTs"
        
        # Debug: Print paths being checked
        print(f"Looking for liver data at: {self.data_root.absolute()}")
        print(f"Images directory: {self.images_dir.absolute()}")
        print(f"Labels directory: {self.labels_dir.absolute()}")
        
        # Check if directories exist
        if not self.data_root.exists():
            print(f"❌ Data root directory not found: {self.data_root.absolute()}")
            self.image_files = []
            self.label_files = []
            return
            
        if not self.images_dir.exists():
            print(f"❌ Images directory not found: {self.images_dir.absolute()}")
            self.image_files = []
            self.label_files = []
            return
            
        if not self.labels_dir.exists():
            print(f"❌ Labels directory not found: {self.labels_dir.absolute()}")
            self.image_files = []
            self.label_files = []
            return
        
        # Get file lists - filter out hidden files
        all_image_files = list(self.images_dir.glob("*.nii.gz"))
        all_label_files = list(self.labels_dir.glob("*.nii.gz"))
        
        print(f"Raw files found - Images: {len(all_image_files)}, Labels: {len(all_label_files)}")
        
        # Remove hidden files (starting with . or _)
        self.image_files = sorted([f for f in all_image_files 
                                  if not f.name.startswith('.') and not f.name.startswith('_')])
        self.label_files = sorted([f for f in all_label_files 
                                  if not f.name.startswith('.') and not f.name.startswith('_')])
        
        print(f"Found {len(self.image_files)} training images")
        print(f"Found {len(self.label_files)} training labels")
        
        # Debug: Print first few filenames if found
        if len(self.image_files) > 0:
            print(f"Sample image files:")
            for i, f in enumerate(self.image_files[:3]):
                print(f"  {i+1}: {f.name}")
        
        if len(self.label_files) > 0:
            print(f"Sample label files:")
            for i, f in enumerate(self.label_files[:3]):
                print(f"  {i+1}: {f.name}")
        
        # Verify matching pairs
        if len(self.image_files) > 0 and len(self.label_files) > 0:
            try:
                image_numbers = [f.name.split('_')[1].split('.')[0] for f in self.image_files]
                label_numbers = [f.name.split('_')[1].split('.')[0] for f in self.label_files] 
                
                if image_numbers != label_numbers:
                    print("⚠️  Warning: Image and label files don't match!")
                    print(f"Image IDs: {image_numbers[:5]}...")
                    print(f"Label IDs: {label_numbers[:5]}...")
                else:
                    print(f"✅ All {len(self.image_files)} image-label pairs matched")
            except Exception as e:
                print(f"⚠️  Warning: Could not verify file pairing: {e}")
                print("Files might have different naming convention")
        
    def find_liver_data_automatically(self):
        """Automatically find Task03_Liver folder"""
        possible_paths = [
            "C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver",
            "..\\Task03_Liver", 
            "..\\..\\Task03_Liver",
            "C:\\Users\\saura\\Task03_Liver",
            "Task03_Liver",
            "C:\\Users\\saura\\Downloads\\Task03_Liver",
            "C:\\Users\\saura\\Desktop\\Task03_Liver"
        ]
        
        for path_str in possible_paths:
            path = Path(path_str)
            if path.exists() and (path / "imagesTr").exists():
                return str(path.absolute())
        return None
        
    def load_volume_and_mask(self, volume_path, mask_path):
        """Load NIfTI volume and corresponding mask"""
        try:
            volume = nib.load(volume_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            return volume, mask
        except Exception as e:
            print(f"Error loading {volume_path}: {e}")
            return None, None
    
    def preprocess_liver_volume(self, volume_path, mask_path, target_size=(64, 64, 64)):
        """
        Preprocess liver volume for 3D autoencoder
        Optimized for liver anatomy and RTX 4050 Laptop GPU
        """
        # Load NIfTI files
        volume, mask = self.load_volume_and_mask(volume_path, mask_path)
        if volume is None:
            return None, None
            
        print(f"Original volume shape: {volume.shape}")
        print(f"Original mask shape: {mask.shape}")
        
        # Find liver bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            print("No liver mask found!")
            return None, None
            
        # Add padding around liver (liver is larger than spleen)
        z_min = max(0, coords[2].min() - 10)
        z_max = min(volume.shape[2], coords[2].max() + 10)
        y_min = max(0, coords[1].min() - 20)
        y_max = min(volume.shape[1], coords[1].max() + 20)
        x_min = max(0, coords[0].min() - 20)
        x_max = min(volume.shape[0], coords[0].max() + 20)
        
        # Crop to liver region
        cropped_volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        cropped_mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
        
        print(f"Cropped volume shape: {cropped_volume.shape}")
        
        # Normalize CT intensities for liver tissue
        # Liver tissue: approximately 50-70 HU
        volume_norm = np.clip(cropped_volume, -100, 200)  # Liver-optimized window
        volume_norm = (volume_norm + 100) / 300  # Normalize to [0,1]
        
        # Resize to standard dimensions for GPU memory management
        zoom_factors = [target_size[i] / volume_norm.shape[i] for i in range(3)]
        volume_resized = ndimage.zoom(volume_norm, zoom_factors, order=1)
        mask_resized = ndimage.zoom(cropped_mask.astype(float), zoom_factors, order=0)
        
        print(f"Final volume shape: {volume_resized.shape}")
        
        return volume_resized, mask_resized

    def visualize_preprocessing(self, idx=0):
        """Visualize liver preprocessing results"""
        if idx >= len(self.image_files):
            print(f"Index {idx} out of range. Max: {len(self.image_files)-1}")
            return
            
        volume_path = self.image_files[idx]
        mask_path = self.label_files[idx]
        
        print(f"Processing: {volume_path.name}")
        
        # Original data
        original_volume, original_mask = self.load_volume_and_mask(volume_path, mask_path)
        
        # Preprocessed data  
        processed_volume, processed_mask = self.preprocess_liver_volume(volume_path, mask_path)
        
        if processed_volume is None:
            return
            
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original data - middle slices
        mid_z_orig = original_volume.shape[2] // 2
        mid_z_proc = processed_volume.shape[2] // 2
        
        axes[0, 0].imshow(original_volume[:, :, mid_z_orig], cmap='gray')
        axes[0, 0].set_title(f'Original Volume\n{original_volume.shape}')
        
        axes[0, 1].imshow(original_mask[:, :, mid_z_orig], cmap='hot')
        axes[0, 1].set_title(f'Original Mask\n{original_mask.shape}')
        
        axes[0, 2].imshow(original_volume[:, :, mid_z_orig] * (original_mask[:, :, mid_z_orig] > 0), cmap='gray')
        axes[0, 2].set_title('Original Liver Region')
        
        # Processed data
        axes[1, 0].imshow(processed_volume[:, :, mid_z_proc], cmap='gray')
        axes[1, 0].set_title(f'Processed Volume\n{processed_volume.shape}')
        
        axes[1, 1].imshow(processed_mask[:, :, mid_z_proc], cmap='hot')
        axes[1, 1].set_title(f'Processed Mask\n{processed_mask.shape}')
        
        axes[1, 2].imshow(processed_volume[:, :, mid_z_proc] * (processed_mask[:, :, mid_z_proc] > 0), cmap='gray')
        axes[1, 2].set_title('Processed Liver Region')
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        plt.savefig('../results/liver_preprocessing_example.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\nIntensity Statistics:")
        print(f"Original - Min: {original_volume.min():.1f}, Max: {original_volume.max():.1f}")
        print(f"Processed - Min: {processed_volume.min():.3f}, Max: {processed_volume.max():.3f}")
        print(f"Liver voxels: {np.sum(processed_mask > 0)}")

# Usage example and testing
if __name__ == "__main__":
    print("🔍 Searching for liver data...")
    
    # Try to find data automatically
    possible_paths = [
        "C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver",
        "..\\Task03_Liver", 
        "..\\..\\Task03_Liver",
        "C:\\Users\\saura\\Task03_Liver",
        "Task03_Liver",
        "C:\\Users\\saura\\Downloads\\Task03_Liver",
        "C:\\Users\\saura\\Desktop\\Task03_Liver"
    ]
    
    data_root = None
    for path_str in possible_paths:
        path = Path(path_str)
        print(f"Checking: {path.absolute()}")
        if path.exists() and (path / "imagesTr").exists():
            data_root = str(path.absolute())
            print(f"✅ Found liver data at: {data_root}")
            break
        else:
            print(f"❌ Not found")
    
    if data_root is None:
        print("\n❌ Task03_Liver folder not found!")
        print("Please ensure the Task03_Liver folder is in one of these locations:")
        for path in possible_paths:
            print(f"  - {Path(path).absolute()}")
        print("\nOr update the path manually in the code.")
        exit(1)
    
    # Initialize preprocessor with found path
    preprocessor = LiverDataPreprocessor(data_root)
    
    # Test preprocessing on first sample
    print("\n=== Testing Liver Preprocessing ===")
    if len(preprocessor.image_files) > 0:
        print(f"✅ Found {len(preprocessor.image_files)} liver images")
        
        try:
            volume_path = preprocessor.image_files[0]
            mask_path = preprocessor.label_files[0]
            print(f"Testing with: {volume_path.name}")
            
            processed_volume, processed_mask = preprocessor.preprocess_liver_volume(volume_path, mask_path)
            
            if processed_volume is not None:
                print(f"✅ Preprocessing successful!")
                print(f"Liver voxels: {np.sum(processed_mask > 0)}")
                print(f"Volume range: {processed_volume.min():.3f} to {processed_volume.max():.3f}")
            else:
                print("❌ Preprocessing failed")
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No liver images found! Please check:")
        print("1. Task03_Liver folder exists")
        print("2. imagesTr/ and labelsTr/ subfolders exist")
        print("3. .nii.gz files are present")
        print(f"4. Looking in: {data_root}")
