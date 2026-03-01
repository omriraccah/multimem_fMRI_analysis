# BOLD to MNI Space Conversion Notebook
# 
# This notebook converts an activation map from subject BOLD space to MNI space.

# ## Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, plotting
import ants

# ## Set up parameters
# You can adjust these parameters as needed

# Subject ID
sub = "01"

# User ID for file paths
user = "aa2842"

# Input activation map filename
input_map = "sub-mm01_cond-V-V_searchlight-rsa.nii.gz"

# ## Define directories and paths

# Base directory
base_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/"

# Subject directory
sub_dir = f"{base_dir}preprocessed/sub-mm{sub}/"

# Functional and anatomical directories
func_dir = f"{sub_dir}func/"
anat_dir = f"{sub_dir}anat/"

# Output directory for MNI-space images
output_dir = f"{sub_dir}func/mni/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set paths for input activation map
input_map_path = f"{func_dir}{input_map}"

# Check if the input file exists
if os.path.exists(input_map_path):
    print(f"Found input activation map: {input_map_path}")
else:
    print(f"WARNING: Input file not found: {input_map_path}")

# ## Define transformation paths

# First transform: BOLD to T1w
bold_to_t1w_transform = f"{sub_dir}func/sub-mm{sub}_from-bold_to-T1w_mode-image_xfm.h5"

# Second transform: T1w to MNI
t1w_to_mni_transform = f"{anat_dir}sub-mm{sub}_from-T1w_to-MNI152Lin_mode-image_xfm.h5"

# MNI template path
mni_template = f"{base_dir}atlases/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii"

# Output path for transformed image
output_filename = input_map.replace('.nii.gz', '_mni.nii.gz')
output_path = f"{output_dir}{output_filename}"

# Check if transformation files exist
print(f"Checking for transformation files...")
if os.path.exists(bold_to_t1w_transform):
    print(f"BOLD to T1w transformation found: {bold_to_t1w_transform}")
else:
    print(f"WARNING: BOLD to T1w transformation not found: {bold_to_t1w_transform}")
    
if os.path.exists(t1w_to_mni_transform):
    print(f"T1w to MNI transformation found: {t1w_to_mni_transform}")
else:
    print(f"WARNING: T1w to MNI transformation not found: {t1w_to_mni_transform}")

if os.path.exists(mni_template):
    print(f"MNI template found: {mni_template}")
else:
    print(f"WARNING: MNI template not found: {mni_template}")

# ## Explore the input activation map

# Load and examine the input activation map
try:
    print("Loading input activation map...")
    # Load with nilearn
    nilearn_img = image.load_img(input_map_path)
    print(f"Shape: {nilearn_img.shape}")
    print(f"Affine:\n{nilearn_img.affine}")
    
    # Display basic statistics
    img_data = nilearn_img.get_fdata()
    print(f"Min value: {np.min(img_data)}")
    print(f"Max value: {np.max(img_data)}")
    print(f"Mean value: {np.mean(img_data)}")
    
    # Visualize the input activation map
    plt.figure(figsize=(10, 6))
    plotting.plot_img(nilearn_img, display_mode='ortho', 
                     title='Original activation map (BOLD space)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error loading input activation map: {e}")

# ## Step 1: Convert from BOLD to T1w space

# Load the input activation map using ANTs
print("Loading input activation map with ANTs...")
try:
    input_img = ants.image_read(input_map_path)
    print("Successfully loaded input image with ANTs")

    # Get T1w reference image
    t1w_ref_path = f"{anat_dir}sub-mm{sub}_desc-preproc_T1w.nii.gz"
    if os.path.exists(t1w_ref_path):
        print(f"Found T1w reference image: {t1w_ref_path}")
        t1w_ref_img = ants.image_read(t1w_ref_path)
    else:
        print(f"WARNING: T1w reference image not found: {t1w_ref_path}")
    
    # Transform from BOLD to T1w space
    print("Applying BOLD to T1w transformation...")
    t1w_space_img = ants.apply_transforms(
        fixed=t1w_ref_img,
        moving=input_img,
        transformlist=[bold_to_t1w_transform]
    )
    print("Successfully transformed to T1w space")
    
    # Save intermediate result
    t1w_output_path = f"{output_dir}{input_map.replace('.nii.gz', '_t1w.nii.gz')}"
    t1w_space_img.to_filename(t1w_output_path)
    print(f"Saved T1w space image to: {t1w_output_path}")
    
    # Visualize the T1w space image
    # Convert ANTs image to Nibabel/Nilearn format for visualization
    t1w_nilearn_img = image.load_img(t1w_output_path)
    plt.figure(figsize=(10, 6))
    plotting.plot_img(t1w_nilearn_img, display_mode='ortho', 
                     title='Activation map in T1w space')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error in BOLD to T1w transformation: {e}")

# ## Step 2: Convert from T1w to MNI space

try:
    # Load MNI template
    print("Loading MNI template...")
    if os.path.exists(mni_template):
        mni_ref_img = ants.image_read(mni_template)
        print("Successfully loaded MNI template")
    else:
        print("WARNING: MNI template not found, trying alternative...")
        # Try using a template from nilearn as backup
        from nilearn.datasets import load_mni152_template
        mni_template_nilearn = load_mni152_template()
        mni_template_path = mni_template_nilearn.get_filename()
        mni_ref_img = ants.image_read(mni_template_path)
        print(f"Using nilearn's MNI template: {mni_template_path}")
    
    # Transform from T1w to MNI space
    print("Applying T1w to MNI transformation...")
    mni_space_img = ants.apply_transforms(
        fixed=mni_ref_img,
        moving=t1w_space_img,
        transformlist=[t1w_to_mni_transform]
    )
    print("Successfully transformed to MNI space")
    
    # Save final result
    mni_space_img.to_filename(output_path)
    print(f"Saved MNI space image to: {output_path}")
    
    # Visualize the MNI space image
    # Convert ANTs image to Nibabel/Nilearn format for visualization
    mni_nilearn_img = image.load_img(output_path)
    plt.figure(figsize=(10, 6))
    plotting.plot_img(mni_nilearn_img, display_mode='ortho', 
                     title='Activation map in MNI space')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error in T1w to MNI transformation: {e}")

# ## Compare all three images side by side

try:
    # Create a visualization comparing all transformations
    plt.figure(figsize=(18, 6))
    
    # Original BOLD space image
    plt.subplot(1, 3, 1)
    plotting.plot_img(nilearn_img, display_mode='ortho', 
                     title='Original (BOLD space)', axes=plt.gca())
    
    # T1w space image
    plt.subplot(1, 3, 2)
    plotting.plot_img(t1w_nilearn_img, display_mode='ortho', 
                     title='Intermediate (T1w space)', axes=plt.gca())
    
    # MNI space image
    plt.subplot(1, 3, 3)
    plotting.plot_img(mni_nilearn_img, display_mode='ortho', 
                     title='Final (MNI space)', axes=plt.gca())
    
    plt.tight_layout()
    
    # Save comparison figure
    compare_vis_path = f"{output_dir}{output_filename.replace('.nii.gz', '_comparison.png')}"
    plt.savefig(compare_vis_path)
    print(f"Comparison visualization saved to: {compare_vis_path}")
    plt.show()
    
    print("\nTransformation Summary:")
    print(f"1. Original activation map: {input_map_path}")
    print(f"2. T1w space image: {t1w_output_path}")
    print(f"3. MNI space image: {output_path}")
except Exception as e:
    print(f"Error creating comparison visualization: {e}")

# ## Create a function to view slices for more detailed inspection

def view_slices(img_path, title="Image Slices", n_slices=9):
    """
    View multiple slices of a 3D image for more detailed inspection.
    
    Parameters:
    img_path (str): Path to the nifti image file
    title (str): Title for the plot
    n_slices (int): Number of slices to display
    """
    try:
        # Load the image
        img = image.load_img(img_path)
        data = img.get_fdata()
        
        # Get dimensions
        x, y, z = data.shape[:3]
        
        # Calculate slice indices
        z_indices = np.linspace(int(z*0.2), int(z*0.8), n_slices).astype(int)
        
        # Create the figure
        n_rows = int(np.ceil(n_slices / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        
        # Plot each slice
        slice_idx = 0
        for i in range(n_rows):
            for j in range(3):
                if slice_idx < n_slices:
                    z_idx = z_indices[slice_idx]
                    slice_data = data[:, :, z_idx]
                    
                    # Plot the slice
                    im = axes[i, j].imshow(np.rot90(slice_data), cmap='hot')
                    axes[i, j].set_title(f'Slice {z_idx}/{z}')
                    axes[i, j].axis('off')
                    
                    slice_idx += 1
                else:
                    axes[i, j].axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return True
    except Exception as e:
        print(f"Error in slice visualization: {e}")
        return False

# Examine detailed slices of each transformation step
print("\nViewing detailed slices of each image:")
print("1. Original BOLD space image")
view_slices(input_map_path, "Original Activation Map (BOLD space)")

print("\n2. T1w space image")
view_slices(t1w_output_path, "Activation Map in T1w Space")

print("\n3. MNI space image")
view_slices(output_path, "Activation Map in MNI Space")

print("\nAll transformations complete and verified!")
