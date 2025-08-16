import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# -------------------------------
# Load Plant Disease Model (.h5)
# -------------------------------
def load_plant_model(model_path: str):
    """
    Load a Keras H5 model for plant disease detection.
    """
    model = load_model(model_path)
    return model


# -------------------------------
# Load Soil Model (.tflite)
# -------------------------------
def load_soil_model(model_path: str):
    """
    Load a TFLite model for soil classification.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# -------------------------------
# Preprocess Frame
# -------------------------------
def preprocess_frame(frame, target_size=(224, 224)):
    """
    Resize and normalize a frame for model input.
    """
    img = cv2.resize(frame, target_size)
    img = img.astype("float32") / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)    # add batch dimension
    return img


# -------------------------------
# Predict Plant Disease
# -------------------------------
def predict_plant(frame, model, labels):
    """
    Run prediction for plant disease model.
    """
    img = preprocess_frame(frame, target_size=(224, 224))
    preds = model.predict(img, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = labels[class_idx]
    return label, confidence


# -------------------------------
# Predict Soil Type
# -------------------------------
def predict_soil(frame, interpreter, labels):
    """
    Run prediction for soil classification model.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    target_height = input_details[0]['shape'][1]
    target_width = input_details[0]['shape'][2]

    img = preprocess_frame(frame, target_size=(target_width, target_height))

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    class_idx = int(np.argmax(output_data))
    confidence = float(output_data[class_idx])
    label = labels[class_idx]
    return label, confidence
