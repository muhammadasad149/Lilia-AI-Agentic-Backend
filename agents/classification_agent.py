import numpy as np
import tensorflow as tf
from agents.preprocessing_agent import preprocess_nii

# Load model only once for efficiency
MODEL_PATH = "model/classification_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def model_predict(nii_path):
    processed_img = preprocess_nii(nii_path)
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(result))
    confidence = float(np.max(result)) * 100
    class_label = "Aneurysm Detected" if predicted_class == 1 else "Aneurysm Not Detected"
    return class_label, confidence
