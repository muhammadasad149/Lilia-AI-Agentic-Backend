import nibabel as nib
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = './model/segmentation_model.h5'
model = load_model(MODEL_PATH, compile=False)

def load_nii_slices(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    return data  # shape: (H, W, Slices)

def predict_on_every_7th_slice(nii_data, model, img_size=(256, 256)):
    preds = []
    indices = []
    for i in range(0, nii_data.shape[2], 7):  # Every 7th slice
        slice_ = nii_data[:, :, i]
        # Resize and normalize
        slice_resized = cv2.resize(slice_, img_size)
        slice_norm = (slice_resized - np.min(slice_resized)) / (np.max(slice_resized) - np.min(slice_resized) + 1e-7)
        # Expand dimensions for model input
        input_img = np.expand_dims(slice_norm, axis=(0, -1)).astype(np.float32)
        # Predict and binarize
        pred_mask = model.predict(input_img, verbose=0)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        preds.append(pred_mask)
        indices.append(i)
    return np.array(preds), indices

def create_region_and_overlapped_images(nii_data, masks, indices):
    import base64
    from io import BytesIO
    from PIL import Image

    region_images = []
    overlapped_images = []
    for j, i in enumerate(indices):
        # Region image (white mask on black)
        mask_img = (masks[j] * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_img)
        buf1 = BytesIO()
        mask_pil.save(buf1, format="PNG")
        region_images.append(f"data:image/png;base64,{base64.b64encode(buf1.getvalue()).decode('utf-8')}")

        # Overlapped image (original + white mask)
        orig_slice = nii_data[:, :, i]
        orig_resized = cv2.resize(orig_slice, (256, 256))
        orig_norm = cv2.normalize(orig_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlapped = np.stack([orig_norm]*3, axis=-1)
        # Draw mask in white where pred_mask==1
        overlapped[masks[j]==1] = [255,255,255]  # white color
        overlapped_pil = Image.fromarray(overlapped)
        buf2 = BytesIO()
        overlapped_pil.save(buf2, format="PNG")
        overlapped_images.append(f"data:image/png;base64,{base64.b64encode(buf2.getvalue()).decode('utf-8')}")
    return region_images, overlapped_images

def segmentation_pipeline(nii_path):
    nii_data = load_nii_slices(nii_path)
    predicted_masks, selected_indices = predict_on_every_7th_slice(nii_data, model, img_size=(256, 256))
    region_imgs, overlapped_imgs = create_region_and_overlapped_images(nii_data, predicted_masks, selected_indices)
    return {
        "region_images": region_imgs,
        "overlapped_images": overlapped_imgs,
        "slice_indices": selected_indices
    }
