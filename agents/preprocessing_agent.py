import nibabel as nib
import numpy as np
import cv2

def preprocess_nii(nii_path, img_size=(256, 256)):
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    mid_slice_idx = nii_data.shape[2] // 2
    mid_slice = nii_data[:, :, mid_slice_idx]
    normalized_slice = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX)
    resized_slice = cv2.resize(normalized_slice, img_size)
    final_img = np.stack([resized_slice] * 3, axis=-1)
    final_img = final_img / 255.0 
    final_img = np.expand_dims(final_img, axis=0)
    return final_img.astype(np.float32)
