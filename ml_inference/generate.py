import qrcode
import qrcode.image.svg # For SVG generation
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import string
import math
from pyzbar.pyzbar import decode as pyzbar_decode # For QR code scannability check

# --- Configuration ---
NUM_QR_CODES = 5000  # Increased number of QR codes to generate
IMAGE_SIZE = (128, 128) # Target pixel size for PNGs for LSB (width, height)
# LSB_BITS_PER_CHANNEL will now be dynamically chosen per image
DATA_DIRECTORY = 'upi_qr_stego_dataset_lsb'
METADATA_FILE = os.path.join(DATA_DIRECTORY, 'metadata', 'dataset_metadata.csv')

# Ensure output directories exist
os.makedirs(os.path.join(DATA_DIRECTORY, 'clean_png'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIRECTORY, 'clean_svg'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIRECTORY, 'stego_lsb_png'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIRECTORY, 'metadata'), exist_ok=True)

# --- Helper Functions ---

def generate_random_upi_uri():
    """Generates a random UPI payment URI string."""
    payee_vpas = [
        "merchant123@bankupi",
        "shopkeeper.xyz@upi",
        "foodcorner@pay",
        "electronics.store@upiid",
        "coffee_shop@ybl",
        "grocerymart@okicici",
        "bookstore.online@apl",
        "pharmacy.meds@axisbank",
        "taxi_service@ybl",
        "beautyparlour@paytm"
    ]
    payee_names = [
        "Aadesh Enterprises", "Bhavani Stores", "City Mart", "Digital Solutions", 
        "Express Foods", "Global Traders", "Happy Home Appliances", "Ideal Books",
        "Jumbo Groceries", "Klassic Kicks"
    ]
    transaction_notes = [
        "Payment for order", "Invoice", "Purchase", "Service Fee", "Online Payment", 
        "Goods Delivery", "Rental Payment", "Subscription Renewal", "Donation", "Utility Bill"
    ]

    pa = random.choice(payee_vpas)
    pn = random.choice(payee_names).replace(" ", "%20") # URL encode spaces
    am = f"{random.uniform(10.0, 5000.0):.2f}" # Random amount
    tn = random.choice(transaction_notes).replace(" ", "%20") # URL encode spaces
    tr = ''.join(random.choices(string.digits, k=10)) # Random transaction reference

    # Randomly choose between static and dynamic QR content
    if random.random() < 0.6: # 60% chance for dynamic QR to encourage more variety
        upi_uri = f"upi://pay?pa={pa}&pn={pn}&am={am}&cu=INR&tn={tn}&tr={tr}"
        qr_type = "Dynamic"
    else:
        upi_uri = f"upi://pay?pa={pa}&pn={pn}&tn={tn}"
        qr_type = "Static"
    
    return upi_uri, qr_type, pa, pn.replace("%20", " "), am if qr_type == "Dynamic" else None, tn.replace("%20", " "), tr if qr_type == "Dynamic" else None

def generate_random_hidden_message(length_bits):
    """Generates a random binary string of specified length."""
    return ''.join(random.choice('01') for _ in range(length_bits))

def calculate_psnr_mse(original_img_array, stego_img_array):
    """Calculates PSNR and MSE between two images."""
    # Ensure images are float32 for calculation
    original_img_array = original_img_array.astype(np.float32)
    stego_img_array = stego_img_array.astype(np.float32)

    mse = np.mean((original_img_array - stego_img_array) ** 2)
    if mse == 0:
        return 100.0, 0.0 # PSNR is infinite, set to a high value, MSE is 0
    max_pixel_value = 255.0 # For 8-bit images
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr, mse

def hide_message_lsb(image_array, binary_message, num_bits_per_channel=1):
    """
    Hides a binary message in the least significant bits of an image array.
    Assumes image_array is uint8 (0-255).
    """
    stego_image_array = image_array.copy()
    message_idx = 0
    message_len = len(binary_message)
    
    # Create a mask to clear the LSBs
    # If num_bits_per_channel is 1, mask is 0b11111110 (254)
    # If num_bits_per_channel is 2, mask is 0b11111100 (252)
    clear_mask = ~((1 << num_bits_per_channel) - 1) & 0xFF

    height, width, channels = stego_image_array.shape

    for row in range(height):
        for col in range(width):
            for ch in range(channels):
                if message_idx < message_len:
                    # Get the bits from the message
                    # Ensure we don't try to read beyond the message length
                    end_idx = min(message_idx + num_bits_per_channel, message_len)
                    bits_to_hide_str = binary_message[message_idx : end_idx]
                    
                    # Pad with zeros if the last segment is shorter than num_bits_per_channel
                    if len(bits_to_hide_str) < num_bits_per_channel:
                        bits_to_hide_str = bits_to_hide_str.ljust(num_bits_per_channel, '0')

                    bits_to_hide = int(bits_to_hide_str, 2)
                    
                    # Clear the LSBs of the pixel and set new bits
                    current_pixel_value = stego_image_array[row, col, ch]
                    stego_image_array[row, col, ch] = (current_pixel_value & clear_mask) | bits_to_hide
                    
                    message_idx += num_bits_per_channel
                else:
                    return stego_image_array # Message fully embedded

    return stego_image_array

def decode_qr_code(image_path):
    """Decodes a QR code image and returns its data."""
    try:
        data = pyzbar_decode(Image.open(image_path))
        if data:
            return data[0].data.decode('utf-8')
        return None
    except Exception as e:
        # print(f"Error decoding {image_path}: {e}") # Suppress frequent error prints
        return None

# --- Main Dataset Generation Loop ---
metadata_records = []

print(f"Starting dataset generation for {NUM_QR_CODES} QR codes...")

for i in range(NUM_QR_CODES):
    image_id_base = f"qr_{i:05d}"
    
    # 1. Generate UPI URI and QR code properties
    upi_uri, qr_type, payee_vpa, payee_name, amount, transaction_note, transaction_reference = generate_random_upi_uri()
    
    # Random QR code generation parameters
    # Keep version lower for smaller images that get affected more by LSB
    qr_version = random.randint(1, 7) # Adjusted max version for more consistent behavior
    error_correction_level = random.choice([qrcode.constants.ERROR_CORRECT_L,
                                            qrcode.constants.ERROR_CORRECT_M,
                                            qrcode.constants.ERROR_CORRECT_Q,
                                            qrcode.constants.ERROR_CORRECT_H])
    # Adjust box_size to make sure IMAGE_SIZE is somewhat respected
    # Formula: (version * 4 + 17) modules for a square QR code.
    # So, box_size = IMAGE_SIZE[0] / (version * 4 + 17)
    # We ensure box_size is at least 1.
    box_size_calc = IMAGE_SIZE[0] // (qr_version * 4 + 17) 
    box_size = max(1, box_size_calc)
    
    border_size = 4 # Minimum border

    # Create QR code instance
    qr = qrcode.QRCode(
        version=qr_version,
        error_correction=error_correction_level,
        box_size=box_size,
        border=border_size,
    )
    qr.add_data(upi_uri)
    qr.make(fit=True)

    # Convert error correction level constant to string for metadata
    ec_level_str = {
        qrcode.constants.ERROR_CORRECT_L: 'L',
        qrcode.constants.ERROR_CORRECT_M: 'M',
        qrcode.constants.ERROR_CORRECT_Q: 'Q',
        qrcode.constants.ERROR_CORRECT_H: 'H',
    }[error_correction_level]

    # --- 2. Generate Clean QR Code (PNG and SVG) ---
    clean_png_image_id = f"{image_id_base}_clean.png"
    clean_svg_image_id = f"{image_id_base}_clean.svg"
    clean_png_path = os.path.join(DATA_DIRECTORY, 'clean_png', clean_png_image_id)
    clean_svg_path = os.path.join(DATA_DIRECTORY, 'clean_svg', clean_svg_image_id)

    # Generate PNG image
    img_png = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    # Resize to target for consistency. Use LANCZOS for high quality downsampling.
    img_png = img_png.resize(IMAGE_SIZE, Image.Resampling.LANCZOS) 
    img_png.save(clean_png_path)
    
    # Generate SVG image
    img_svg = qr.make_image(image_factory=qrcode.image.svg.SvgImage)
    with open(clean_svg_path, 'wb') as f:
        # Corrected line: removed .encode('utf-8') as to_string() already returns bytes
        f.write(img_svg.to_string()) 

    # Record metadata for clean QR code
    metadata_records.append({
        'image_id': clean_png_image_id,
        'file_path': os.path.join('clean_png', clean_png_image_id),
        'image_properties_resolution_width': IMAGE_SIZE[0],
        'image_properties_resolution_height': IMAGE_SIZE[1],
        'image_properties_file_format': 'PNG',
        'image_properties_color_depth': 24, # RGB
        'qr_code_properties_qr_version': qr_version,
        'qr_code_properties_error_correction_level': ec_level_str,
        'qr_code_properties_box_size': box_size,
        'qr_code_properties_border_size': border_size,
        'qr_code_properties_fill_color': 'black',
        'qr_code_properties_back_color': 'white',
        'upi_data_upi_uri_string': upi_uri,
        'upi_data_payee_vpa': payee_vpa,
        'upi_data_payee_name': payee_name,
        'upi_data_amount': amount,
        'upi_data_transaction_note': transaction_note,
        'upi_data_transaction_reference': transaction_reference,
        'upi_data_qr_code_type': qr_type,
        'steganography_status': False,
        'stego_technique': None,
        'hidden_message': None,
        'embedding_parameters': None,
        'scannability_post_stego': True, # Clean codes are expected to be scannable
        'psnr': None,
        'mse': None,
        'corresponding_clean_png_id': clean_png_image_id # Self-reference for clean
    })
    # Also add a record for the SVG version (optional, but good for completeness)
    metadata_records.append({
        'image_id': clean_svg_image_id,
        'file_path': os.path.join('clean_svg', clean_svg_image_id),
        'image_properties_resolution_width': None, # SVG is vector, no fixed pixel resolution
        'image_properties_resolution_height': None,
        'image_properties_file_format': 'SVG',
        'image_properties_color_depth': None,
        'qr_code_properties_qr_version': qr_version,
        'qr_code_properties_error_correction_level': ec_level_str,
        'qr_code_properties_box_size': box_size,
        'qr_code_properties_border_size': border_size,
        'qr_code_properties_fill_color': 'black',
        'qr_code_properties_back_color': 'white',
        'upi_data_upi_uri_string': upi_uri,
        'upi_data_payee_vpa': payee_vpa,
        'upi_data_payee_name': payee_name,
        'upi_data_amount': amount,
        'upi_data_transaction_note': transaction_note,
        'upi_data_transaction_reference': transaction_reference,
        'upi_data_qr_code_type': qr_type,
        'steganography_status': False,
        'stego_technique': None,
        'hidden_message': None,
        'embedding_parameters': None,
        'scannability_post_stego': True,
        'psnr': None,
        'mse': None,
        'corresponding_clean_png_id': clean_png_image_id
    })


    # --- 3. Apply LSB Modulation ---
    # Convert PIL image to numpy array for LSB manipulation
    original_img_array = np.array(img_png) # This is already RGB from .convert('RGB')
    
    # Randomly choose LSB_BITS_PER_CHANNEL for this stego image
    # Weights are used to make some LSB depths more common than others
    # Example: 1-bit is most common, then 2-bit, less common 3-bit, and 4-bit (more likely to break)
    lsb_bits_choices = [1, 2, 3, 4]
    # Adjust weights based on desired distribution of "Safe", "Suspicious", "Malicious"
    # More weight on higher bits for more malicious examples, but keep 1-bit as common for suspicious
    lsb_bits_weights = [0.4, 0.3, 0.2, 0.1] # Sum to 1.0

    selected_lsb_bits = random.choices(lsb_bits_choices, weights=lsb_bits_weights, k=1)[0]

    # Calculate maximum LSB capacity
    max_capacity_bits = original_img_array.shape[0] * original_img_array.shape[1] * original_img_array.shape[2] * selected_lsb_bits
    
    # Keep message length well within capacity to avoid issues, e.g., 50-75%
    # Adjust range to make it more likely to fill up and cause more distortion
    hidden_message_length_bits = random.randint(int(max_capacity_bits * 0.5), int(max_capacity_bits * 0.75))
    hidden_message = generate_random_hidden_message(hidden_message_length_bits)

    # Apply LSB embedding
    stego_img_array = hide_message_lsb(original_img_array, hidden_message, selected_lsb_bits)
    stego_pil_image = Image.fromarray(stego_img_array.astype(np.uint8))

    # Include the LSB bit depth in the stego image ID for clarity
    stego_png_image_id = f"{image_id_base}_stego_lsb_{selected_lsb_bits}bit.png"
    stego_png_path = os.path.join(DATA_DIRECTORY, 'stego_lsb_png', stego_png_image_id)
    stego_pil_image.save(stego_png_path)

    # Calculate PSNR and MSE
    psnr, mse = calculate_psnr_mse(original_img_array, stego_img_array)

    # Test scannability of the stego QR code
    scanned_data = decode_qr_code(stego_png_path)
    stego_scannable = True if scanned_data == upi_uri else False
    
    # Record metadata for stego QR code
    metadata_records.append({
        'image_id': stego_png_image_id,
        'file_path': os.path.join('stego_lsb_png', stego_png_image_id),
        'image_properties_resolution_width': IMAGE_SIZE[0],
        'image_properties_resolution_height': IMAGE_SIZE[1],
        'image_properties_file_format': 'PNG',
        'image_properties_color_depth': 24, # RGB
        'qr_code_properties_qr_version': qr_version,
        'qr_code_properties_error_correction_level': ec_level_str,
        'qr_code_properties_box_size': box_size,
        'qr_code_properties_border_size': border_size,
        'qr_code_properties_fill_color': 'black',
        'qr_code_properties_back_color': 'white',
        'upi_data_upi_uri_string': upi_uri,
        'upi_data_payee_vpa': payee_vpa,
        'upi_data_payee_name': payee_name,
        'upi_data_amount': amount,
        'upi_data_transaction_note': transaction_note,
        'upi_data_transaction_reference': transaction_reference,
        'upi_data_qr_code_type': qr_type,
        'steganography_status': True,
        'stego_technique': f"LSB_{selected_lsb_bits}bit_per_channel", # Updated
        'hidden_message': hidden_message,
        'embedding_parameters': f"LSB_depth={selected_lsb_bits}", # Updated
        'scannability_post_stego': stego_scannable,
        'psnr': psnr,
        'mse': mse,
        'corresponding_clean_png_id': clean_png_image_id
    })

    if (i + 1) % 500 == 0: # Print progress every 500 codes
        print(f"Generated {i + 1}/{NUM_QR_CODES} QR codes...")

print("\nDataset generation complete.")

# Create and save metadata DataFrame
metadata_df = pd.DataFrame(metadata_records)
metadata_df.to_csv(METADATA_FILE, index=False)
print(f"Metadata saved to {METADATA_FILE}")

print("\nDataset structure:")
print(f"- {DATA_DIRECTORY}/")
print(f"  - clean_png/ (Original QR codes as PNGs)")
print(f"  - clean_svg/ (Original QR codes as SVGs)")
print(f"  - stego_lsb_png/ (LSB-modulated QR codes as PNGs)")
print(f"  - metadata/dataset_metadata.csv (Comprehensive metadata)")

