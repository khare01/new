import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import hashlib
import json
import os
import time
import uuid

def hash_key(key):
    # Use SHA-256 hash for fixed-length key
    hashed_key = hashlib.sha256(key).digest()[:16]  # Use first 16 bytes for AES-128
    return hashed_key

def generate_uuid():
    # Generate a UUID
    return str(uuid.uuid4())

def pad_pkcs7(s):
    block_size = AES.block_size
    padding = block_size - len(s) % block_size
    return s + bytes([padding]) * padding

def extract_facial_features(image):
    # Implement facial feature extraction if needed
    # You can use libraries like dlib or OpenCV for facial feature extraction
    # Return a compact representation of facial features
    return image

def encrypt_with_aes(data, key):
    # Generate a random initialization vector (IV)
    iv = get_random_bytes(16)

    # Create AES cipher object
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Pad the data and then encrypt
    encrypted_data = iv + cipher.encrypt(pad_pkcs7(data.tobytes()))

    return encrypted_data

def decrypt_with_aes(encrypted_data, key):
    # Extract IV from the encrypted data
    iv = encrypted_data[:16]

    # Create AES cipher object
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the data and remove padding
    decrypted_data = cipher.decrypt(encrypted_data[16:])
    decrypted_data = decrypted_data[:-decrypted_data[-1]]

    return decrypted_data
    
def save_encrypted_data(encrypted_data, image_details, output_path):
    with open(output_path, 'wb') as file:
        file.write(json.dumps(image_details).encode('utf-8') + encrypted_data)

def remove_original_image(image_path):
    # Remove the original image
    os.remove(image_path)

def capture_live_image(scan_duration=10):
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Load the pre-trained Haarcascades face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a window to display the live feed
    cv2.namedWindow("Face Scan", cv2.WINDOW_NORMAL)

    # Initialize variables for capturing frames
    start_time = time.time()
    frames = []

    # Initialize variables for "Scanning..." message animation
    show_message = True
    message_timer = time.time()

    while (time.time() - start_time) < scan_duration:
        # Capture a frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(70, 70))


        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the "Scanning..." message with animation
        if time.time() - message_timer > 0.5:
            show_message = not show_message
            message_timer = time.time()

        if show_message:
            cv2.putText(frame, "Scanning...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame in the "Face Scan" window
        cv2.imshow("Face Scan", frame)

        # Append the frame to the list if a face is detected
        if len(faces) > 0:
            frames.append(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy the window
    cap.release()
    cv2.destroyAllWindows()

    # Check if a face was detected
    if len(frames) > 0:
        # Combine captured frames into a single image
        combined_image = np.mean(frames, axis=0, dtype=np.uint8)

        return combined_image
    else:
        print("Error: No face detected.")
        return None

def main():
    # User-provided image path
    user_image_path = input("Enter the path of the image to encrypt: ")

    # Load the user-provided image
    user_image = cv2.imread(user_image_path)

    # Capture live image for key generation
    print("Please look at the camera for face scanning...")
    live_image = capture_live_image()

    if live_image is not None:
        # Resize live image to match the user image dimensions
        live_image = cv2.resize(live_image, (user_image.shape[1], user_image.shape[0]))

        # Extract facial features and use them as a key
        key = extract_facial_features(live_image)

        if key is not None:
            # Hash the key to get a fixed-length key for AES
            hashed_key = hash_key(key)

            # Generate a UUID for better encryption
            uuid_key = generate_uuid()

            # Get image details
            height, width, channels = user_image.shape

            # Create a dictionary with image details
            image_details = {
                'height': height,
                'width': width,
                'channels': channels,
                'uuid': uuid_key
            }

            # Save encrypted data along with image details
            output_path = user_image_path.replace('.jpg', '_encrypted.enc')  # Change file extension as needed
            encrypted_data = encrypt_with_aes(user_image, hashed_key)
            save_encrypted_data(encrypted_data, image_details, output_path)

            print(f"Encryption completed. Encrypted data and image details saved at {output_path}")

            # Remove the original image
            remove_original_image(user_image_path)
    decrypted_data = decrypt_with_aes(encrypted_data, hashed_key)

    # Print the shape of the decrypted data
    print("Shape of decrypted data:", np.frombuffer(decrypted_data, dtype=np.uint8).shape)

    # Convert the decrypted data back to the original image
    original_image = np.frombuffer(decrypted_data, dtype=np.uint8).reshape(
        image_details['height'], image_details['width'], image_details['channels']
    )

if __name__ == "__main__":
    main()
