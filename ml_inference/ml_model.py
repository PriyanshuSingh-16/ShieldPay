import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight # For calculating class weights
import matplotlib.pyplot as plt

# --- 1. Dataset Loading and Preprocessing ---
# This section assumes you have already run the 'qr_stego_dataset_generator'
# and the dataset is available in the specified DATA_DIRECTORY.

def load_and_label_dataset(data_dir, metadata_file, image_size=(128, 128)):
    """
    Loads QR code images and assigns multi-class labels based on metadata.

    Args:
        data_dir (str): Path to the root directory containing 'clean_png' and 'stego_lsb_png'.
        metadata_file (str): Path to the dataset metadata CSV file.
        image_size (tuple): Target pixel size for images (width, height).

    Returns:
        tuple: A tuple containing:
            - images (np.array): Processed image data.
            - labels (np.array): Corresponding integer labels (0=Safe, 1=Suspicious, 2=Malicious).
    """
    print(f"Loading metadata from: {metadata_file}")
    metadata_df = pd.read_csv(metadata_file)

    images = []
    labels = []
    
    print("Starting image loading and multi-class labeling...")
    # Filter for PNG images only, as LSB applies to raster images
    png_metadata_df = metadata_df[metadata_df['image_properties_file_format'] == 'PNG'].copy()

    for index, row in png_metadata_df.iterrows():
        image_id = row['image_id']
        stego_status = row['steganography_status']
        scannability_post_stego = row['scannability_post_stego']
        file_path_relative = row['file_path'] # e.g., 'clean_png/qr_00000_clean.png'

        # Ensure we only load from 'clean_png' or 'stego_lsb_png' directories
        # based on the relative path in the metadata.
        # This handles cases where file_path_relative might include 'clean_svg'
        # which we explicitly want to skip for this pixel-based model.
        if 'clean_svg' in file_path_relative:
            continue # Skip SVG files

        image_path = os.path.join(data_dir, file_path_relative)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}. Skipping.")
            continue

        try:
            img = Image.open(image_path).convert('RGB') # Ensure 3 channels for RGB
            img = img.resize(image_size, Image.Resampling.LANCZOS) # Resize to consistent target size
            img_array = np.array(img).astype('float32') / 255.0 # Normalize pixel values

            images.append(img_array)

            # Assign multi-class labels
            if not stego_status:
                labels.append(0) # 0: Safe (Clean)
            elif stego_status and scannability_post_stego:
                labels.append(1) # 1: Suspicious (Stego but Scannable)
            else: # stego_status and not scannability_post_stego
                labels.append(2) # 2: Malicious (Stego and Not Scannable)

        except Exception as e:
            print(f"Error loading or processing image {image_id}: {e}")
            continue

    if not images:
        print("No PNG images were loaded. Check paths, data organization, and metadata content.")
        return np.array([]), np.array([])

    return np.array(images), np.array(labels)

# --- 2. Define the Multi-Class CNN Model Architecture ---

def build_multiclass_steganalysis_model(input_shape, num_classes):
    """
    Builds a Convolutional Neural Network (CNN) model for multi-class steganography detection.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes (e.g., 3 for Safe, Suspicious, Malicious).

    Returns:
        tf.keras.Model: The compiled multi-class CNN model.
    """
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output for the Dense layers
        layers.Flatten(),

        # Dense Layer 1
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Dropout for regularization

        # Output Layer (Multi-class Classification)
        layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])

    # Compile the model
    # Use sparse_categorical_crossentropy because labels are integers (0, 1, 2)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 3. Main Training Workflow ---

if __name__ == "__main__":
    # Define paths (Adjust these to your actual dataset location)
    DATA_DIRECTORY = 'upi_qr_stego_dataset_lsb' 
    METADATA_FILE = os.path.join(DATA_DIRECTORY, 'metadata', 'dataset_metadata.csv') 
    IMAGE_TARGET_SIZE = (128, 128) # Ensure this matches or is compatible with dataset generation

    # Define the number of classes for our problem
    NUM_CLASSES = 3 # 0: Safe, 1: Suspicious, 2: Malicious
    CLASS_NAMES = ["Safe", "Suspicious", "Malicious"]

    print("Attempting to load and label dataset...")
    images, labels = load_and_label_dataset(DATA_DIRECTORY, METADATA_FILE, IMAGE_TARGET_SIZE)

    if images.size == 0:
        print("Exiting: No images loaded. Please ensure your dataset path and metadata are correct, and the dataset generator has been run.")
    else:
        print(f"Loaded {len(images)} images with shape {images.shape}. Labels shape: {labels.shape}")
        
        # Print label distribution to identify potential imbalance
        label_series = pd.Series(labels)
        print(f"\nOverall Label distribution:\n{label_series.value_counts().sort_index()}")

        # Determine input shape for the model
        input_shape = images.shape[1:] 
        print(f"Model input shape: {input_shape}")

        # Build the model
        model = build_multiclass_steganalysis_model(input_shape, NUM_CLASSES)
        model.summary()

        # Split data into training and testing sets
        # Stratify ensures that each class is represented proportionally in train and test sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"\nTraining images: {len(train_images)}, Test images: {len(test_images)}")
        print(f"Training label distribution:\n{pd.Series(train_labels).value_counts().sort_index()}")
        print(f"Test label distribution:\n{pd.Series(test_labels).value_counts().sort_index()}")

        # Calculate class weights for imbalanced datasets
        # This helps the model pay more attention to underrepresented classes
        unique_labels = np.unique(train_labels)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=train_labels
        )
        class_weights = dict(zip(unique_labels, weights))
        print(f"\nCalculated Class Weights: {class_weights}")

        # Define Early Stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=5,          # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
        )


        # Train the model
        print("\nStarting model training...")
        history = model.fit(
            train_images, train_labels,
            epochs=50, # Increased epochs, but EarlyStopping will likely stop it sooner
            batch_size=32, 
            validation_split=0.1, 
            verbose=1,
            callbacks=[early_stopping], # Add EarlyStopping callback
            class_weight=class_weights # Apply class weights
        )
        print("Model training complete.")

        # Evaluate the model on the test set
        print("\nEvaluating model on test set...")
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        # Optional: Save the trained model
        model.save('multiclass_steganography_detector_model.h5')
        print("Model saved as 'multiclass_steganography_detector_model.h5'")

        # Optional: Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Optional: Make predictions on a few test images
        print("\nMaking predictions on a few test images...")
        predictions = model.predict(test_images[:5])
        predicted_classes = np.argmax(predictions, axis=1)

        for i in range(5):
            print(f"Image {i}:")
            print(f"  True Label: {CLASS_NAMES[test_labels[i]]} (Index: {test_labels[i]})")
            print(f"  Predicted Label: {CLASS_NAMES[predicted_classes[i]]} (Index: {predicted_classes[i]})")
            print(f"  Prediction Probabilities: {predictions[i]}")
