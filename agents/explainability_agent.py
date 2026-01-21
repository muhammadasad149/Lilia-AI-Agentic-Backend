import os
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import matplotlib.cm as cm
from openai import AzureOpenAI
from typing import Dict, List, Optional
import json
import nibabel as nib
import tensorflow as tf
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import logging

# Set up logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configure LLM
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class EnhancedExplainabilityAgent:
    def __init__(self):
        self.setup_gradcam_model()
        
    def setup_gradcam_model(self):
        """Setup GradCAM model for explainability"""
        try:
            model_path = "model/classification_model.h5" 
            if os.path.exists(model_path):
                self.gradcam_model = tf.keras.models.load_model(model_path)
                print("Successfully loaded full model for GradCAM.")
            else:
                self.gradcam_model = None
                print("Warning: Full Keras model not found at 'model/classification_model.h5'. GradCAM will be disabled.")
        except Exception as e:
            print(f"Error loading GradCAM model: {e}")
            self.gradcam_model = None

    def generate_gradcam_heatmap(self, image_array: np.ndarray, class_index: int) -> Optional[np.ndarray]:
        """Generate Grad-CAM heatmap for a given image and class index."""
        if self.gradcam_model is None:
            return None
        
        try:
            last_conv_layer = next(l for l in reversed(self.gradcam_model.layers) if isinstance(l, tf.keras.layers.Conv2D))
            grad_model = tf.keras.models.Model([self.gradcam_model.inputs], [last_conv_layer.output, self.gradcam_model.output])

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array)
                loss = predictions[:, class_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            return heatmap
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return None

    def create_gradcam_overlay(self, original_image: np.ndarray, heatmap: np.ndarray) -> str:
        """Create a base64 encoded Grad-CAM overlay image."""
        try:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            if len(original_image.shape) == 2:
                original_image = np.stack([original_image] * 3, axis=-1)

            superimposed_img = heatmap * 0.4 + original_image
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

            img_pil = Image.fromarray(superimposed_img)
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        except Exception as e:
            print(f"Error creating Grad-CAM overlay: {e}")
            return ""

    def analyze_image_statistics(self, nii_path: str) -> Dict:
        """Analyze and return key statistics from the NIfTI image."""
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            return {
                'image_dimensions': nii_data.shape,
                'voxel_size': [round(v, 2) for v in nii_img.header.get_zooms()[:3]],
                'intensity_range': [float(nii_data.min()), float(nii_data.max())],
                'mean_intensity': float(np.mean(nii_data)),
            }
        except Exception as e:
            print(f"Error analyzing image statistics: {e}")
            return {}

    def generate_clinical_explanation(self, prediction: str, confidence: float, patient_data: Dict, image_stats: Dict) -> str:
        """Generate a clinical explanation using Gemini or fallback."""
        try:
            prompt = f"""
            As a radiologist AI assistant, provide a comprehensive clinical explanation for this aneurysm detection result.
            
            PREDICTION DETAILS:
            - Prediction: {prediction}
            - Confidence: {confidence:.2f}%
            
            PATIENT CONTEXT:
            - Age: {patient_data.get('age', 'Unknown')}
            - Gender: {patient_data.get('sex', 'Unknown')}
            - History: {patient_data.get('history', 'Not Provided')}
            
            IMAGE CHARACTERISTICS:
            - Dimensions: {image_stats.get('image_dimensions', 'Unknown')}
            - Voxel Size (mm): {image_stats.get('voxel_size', 'Unknown')}
            
            Please provide in structured HTML:
            1. **Clinical Interpretation**
            2. **Confidence Assessment**
            3. **Key Imaging Features**
            4. **Recommended Next Steps**
            5. **Limitations**
            """
            
            messages = [
                ("system", "You are an expert radiology AI assistant providing clinical explanations. Be thorough, accurate, and use structured HTML."),
                ("human", prompt),
            ]
            ai_msg = llm.invoke(messages)
            logging.info(f"AI Explanation: {ai_msg}")
            import re

            def extract_clean_html(ai_response: str) -> str:
                # Extract content between ```html and ```
                match = re.search(r"```html\s*(.*?)\s*```", ai_response, re.DOTALL)
                if match:
                    html_content = match.group(1)
                    # Clean the HTML content
                    # html_content = html_content.replace("\\n", "")  # Remove literal \n
                    html_content = html_content.replace('\n', "")   # Remove real line breaks
                    html_content = html_content.strip()
                    return html_content
                return "<p>Error: No valid HTML found in response.</p>"
            clean_html = extract_clean_html(ai_msg.content)
            logging.info(f"Clean HTML: {clean_html}")
            return clean_html
        except Exception as e:
            print(f"Error generating LLM explanation: {e}")
            return self.generate_fallback_explanation(prediction, confidence)

    def generate_fallback_explanation(self, prediction: str, confidence: float) -> str:
        """Generate a fallback explanation when the LLM is unavailable."""
        result_html = f"""
        <div class="clinical-explanation">
            <h3>Clinical Interpretation</h3>
            <p><strong>Result:</strong> {prediction} with {confidence:.1f}% confidence</p>
            <h4>Key Points:</h4>
        """
        if prediction == "Aneurysm Detected":
            result_html += """
                <ul>
                    <li>The AI model has identified imaging features consistent with an aneurysm.</li>
                    <li>This finding requires urgent radiologist review and clinical correlation.</li>
                </ul>
                <h4>Recommended Actions:</h4>
                <ul>
                    <li>Immediate radiologist review required.</li>
                    <li>Consider additional imaging (CTA/MRA) if not already performed.</li>
                    <li>Neurosurgical consultation may be warranted.</li>
                </ul>
            """
        else:
            result_html += """
                <ul>
                    <li>The AI model did not identify imaging features consistent with an aneurysm.</li>
                    <li>This result supports the absence of obvious aneurysmal changes on the analyzed slices.</li>
                </ul>
                <h4>Clinical Considerations:</h4>
                <ul>
                    <li>Routine radiologist review is still recommended.</li>
                    <li>Small or atypical aneurysms may not be detected by AI.</li>
                </ul>
            """
        result_html += """
            <p><em>Note: This is an AI-generated interpretation and must be verified by a qualified radiologist.</em></p>
        </div>
        """
        return result_html

    def run(self, nii_path: str, prediction_data: dict, patient_data: dict) -> dict:
        """Main execution with smart slice selection + logging."""
        prediction = prediction_data['prediction']
        raw_confidence = prediction_data['confidence']
        
        try:
            confidence = float(str(raw_confidence).replace('%', ''))
        except ValueError:
            confidence = 0.0

        class_idx = 1 if prediction == "Aneurysm Detected" else 0
        logging.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}%, Class Index: {class_idx}")

        # 1. Image statistics
        image_stats = self.analyze_image_statistics(nii_path)
        logging.info(f"Image stats: {image_stats}")

        # 2. Preprocess slices
        from agents.preprocessing_agent import preprocess_nii
        processed = preprocess_nii(nii_path)
        logging.info(f"Processed image shape: {processed.shape}")

        # 3. Auto-select best slice based on Grad-CAM activation
        best_slice_idx = -1
        best_heatmap = None
        max_activation = -1

        for idx in range(processed.shape[0]):
            input_slice = np.expand_dims(processed[idx], axis=0)
            heatmap = self.generate_gradcam_heatmap(input_slice, class_idx)

            if heatmap is not None:
                activation = np.sum(heatmap)
                if activation > max_activation:
                    best_slice_idx = idx
                    max_activation = activation
                    best_heatmap = heatmap

        if best_slice_idx == -1:
            logging.warning("No valid Grad-CAM heatmaps found.")
            return {
                "gradcam_overlay_b64": "",
                "image_statistics": image_stats,
                "diagnostic_report_html": self.generate_clinical_explanation(prediction, confidence, patient_data, image_stats)
            }

        logging.info(f"Best slice selected: {best_slice_idx} with activation {max_activation:.4f}")

        # 4. Get original slice from NIfTI file
        nii_img = nib.load(nii_path)
        original_slice = nii_img.get_fdata()[:, :, best_slice_idx]
        original_slice_norm = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX)

        gradcam_overlay = self.create_gradcam_overlay(original_slice_norm, best_heatmap)

        # 5. Generate explanation
        explanation_html = self.generate_clinical_explanation(prediction, confidence, patient_data, image_stats)

        return {
            "gradcam_overlay_b64": gradcam_overlay,
            "image_statistics": image_stats,
            "diagnostic_report_html": explanation_html
        }
