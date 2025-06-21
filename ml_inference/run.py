import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration ---
# Path to your saved model file (assuming it was saved as .h5 or SavedModel format)
# If you didn't explicitly save it, the 'multiclass_steganography_detector_model.h5'
# from the previous script might not exist unless you uncommented the save line.
# For simplicity, we assume you've run the training and it completed successfully.
MODEL_PATH = 'multiclass_steganography_detector_model.h5' # Or the path where you saved it
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
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img = img.resize(target_size, Image.Resampling.LANCZOS) # Resize
        img_array = np.array(img).astype('float32') / 255.0 # Convert to numpy and normalize
        
        # Add a batch dimension (model expects input shape like (batch_size, height, width, channels))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- Main Prediction Workflow ---
if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        # Load the trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists at the specified path and was saved correctly.")
        print("If you didn't uncomment `model.save(...)` in ml_model.py, the file might not exist.")
        exit() # Exit if model cannot be loaded

    # --- TODO: Replace with the path to your new QR code image ---
    # Example: you might put a clean QR code here, or a stego one generated earlier
    # You will need to manually provide an image path for testing.
    # For instance, if you generated 5000 QR codes, pick one from
    # upi_qr_stego_dataset_lsb/clean_png/ or upi_qr_stego_dataset_lsb/stego_lsb_png/
    
    # You can pick one of your generated images for testing:
    # sample_image_path = os.path.join('upi_qr_stego_dataset_lsb', 'clean_png', 'qr_0000_clean.png')
    # Or a stego one (adjust the ID and bit depth based on your generated files)
    # sample_image_path = os.path.join('upi_qr_stego_dataset_lsb', 'stego_lsb_png', 'qr_00001_stego_lsb_1bit.png') 
    sample_image_path = os.path.join('upi_qr_stego_dataset_lsb', 'stego_lsb_png', 'qr_00002_stego_lsb_4bit.png') # This might be malicious!

    print(f"\nAttempting to predict for image: {sample_image_path}")
    
    # Preprocess the image
    input_image = preprocess_image(sample_image_path, IMAGE_TARGET_SIZE)

    if input_image is not None:
        print("Image preprocessed. Making prediction...")
        # Make a prediction
        # The model outputs probabilities for each class
        predictions = model.predict(input_image)
        
        # Get the predicted class index (the one with the highest probability)
        predicted_class_index = np.argmax(predictions[0])
        
        # Get the corresponding class name
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        print(f"\nPrediction Results for {sample_image_path}:")
        print(f"  Predicted Class: {predicted_class_name}")
        print(f"  Confidence (Probabilities): {predictions[0]}")
        
        print("\nInterpretation:")
        print(f"  - {CLASS_NAMES[0]}: Probability of being a Safe (Clean) QR code.")
        print(f"  - {CLASS_NAMES[1]}: Probability of being a Suspicious (Stego but Scannable) QR code.")
        print(f"  - {CLASS_NAMES[2]}: Probability of being a Malicious (Stego and Not Scannable) QR code.")
    else:
        print("Could not process image for prediction.")

