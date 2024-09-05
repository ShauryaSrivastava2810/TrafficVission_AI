import cv2
import matplotlib.pyplot as plt
import os

# Directory containing subfolders with images
root_dir = 'idd_dataset_extracted_final'  # Adjust this to the root directory where your images are stored

# Iterate through subdirectories
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Construct the full path to the image file
            image_path = os.path.join(subdir, file)

            # Load and display the image
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                plt.imshow(image)
                plt.title(image_path)  # Show image path as title
                plt.axis('off')
            else:
                print("Error: Unable to read image file:", image_path)
