import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import json # Import json module

# --- Configuration (Must match training configuration) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'multiclass_steganography_detector_model.h5')
IMAGE_TARGET_SIZE = (128, 128) # Must match the size used during training

# Define the classes (must match the order used during training: 0, 1, 2)
CLASS_NAMES = ["Safe", "Suspicious", "Malicious"]

# --- Helper Function for Image Preprocessing ---
def preprocess_image(image_path, target_size):
    """
    Loads and preprocesses a single image for model prediction.
    Must match the preprocessing used during training.
    """
    if not os.path.exists(image_path):
        # Print to stderr for errors, stdout for results
        print(f"Error: Image not found at: {image_path}", file=sys.stderr)
        return None
    
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img = img.resize(target_size, Image.Resampling.LANCZOS) # Resize
        img_array = np.array(img).astype('float32') / 255.0 # Convert to numpy and normalize
        
        # Add a batch dimension (model expects input shape like (batch_size, height, width, channels))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None

# --- Main Prediction Logic ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_model.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path_to_predict = sys.argv[1]

    # Load the trained model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("Ensure 'multiclass_steganography_detector_model.h5' is in the ml_inference directory.", file=sys.stderr)
        sys.exit(1)

    # Preprocess the image
    input_image = preprocess_image(image_path_to_predict, IMAGE_TARGET_SIZE)

    if input_image is not None:
        # Make a prediction
        # Set verbose=0 to suppress the progress bar output from model.predict()
        predictions = model.predict(input_image, verbose=0) 
        
        # Get the predicted class index (the one with the highest probability)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # Output results to stdout as JSON for the Node.js backend to consume
        result = {
            "predictedClass": predicted_class_name,
            "confidence": predictions[0].tolist(), # Convert numpy array to list for JSON
            "imagePath": image_path_to_predict # Return the path for reference
        }
        print(json.dumps(result))
    else:
        sys.exit(1) # Indicate failure

