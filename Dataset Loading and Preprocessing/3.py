import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Directory containing the images
leftimg_dir = 'C:/Users/ASUS/Downloads/idd-lite/idd20k_lite/leftImg'  # Adjust this path as needed
output_dir = 'preprocessed_leftimg'  # Directory to save preprocessed images
os.makedirs(output_dir, exist_ok=True)

# Image size to resize to
IMG_SIZE = (224, 224)


# Function to enhance image contrast using CLAHE
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    enhanced_image = cv2.merge([cl1, cl1, cl1])
    return enhanced_image


# Initialize an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in new pixels with the nearest pixel values
)

# Iterate through the 'leftImg' directory to preprocess images
for subdir, dirs, files in os.walk(leftimg_dir):
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Construct the full path to the image file
            image_path = os.path.join(subdir, file)

            # Load the image using OpenCV
            image = cv2.imread(image_path)

            # Check if the image was successfully loaded
            if image is not None:
                # Enhance the image contrast
                enhanced_image = enhance_image(image)

                # Resize the image
                resized_image = cv2.resize(enhanced_image, IMG_SIZE)

                # Save the preprocessed image
                save_path = os.path.join(output_dir, os.path.relpath(image_path, leftimg_dir))
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_path, resized_image)

                # For demonstration purposes: Apply data augmentation and display images
                x = np.expand_dims(resized_image, axis=0)
                it = datagen.flow(x, batch_size=1)
                augmented_images = [next(it)[0] for i in range(3)]

                # Display the original, enhanced, and augmented images
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 4, 1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                plt.title("Enhanced Image")
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                plt.title("Resized Image")
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(augmented_images[0])
                plt.title("Augmented Image")
                plt.axis('off')

                plt.show()
                plt.close('all')  # Close all open figures to free memory
            else:
                print("Error: Unable to read image file:", image_path)

print("Image preprocessing complete.")
