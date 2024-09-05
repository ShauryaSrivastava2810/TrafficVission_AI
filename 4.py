import os
import shutil
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Base directory containing train, test, val
base_dir = 'idd_dataset_extracted/idd20k_lite/leftImg8bit'

# Create new folders for traffic and no_traffic within train, test, and val
subsets = ['val']
categories = ['traffic', 'no_traffic']

for subset in subsets:
    for category in categories:
        new_dir = os.path.join(base_dir, subset, category)
        os.makedirs(new_dir, exist_ok=True)

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transform to preprocess images
transform = T.Compose([T.ToTensor()])


def classify_image(image_path):
    """
    Classify the image based on the number of cars detected.
    If 3 or more cars are detected => 'traffic'
    If less than 3 cars are detected => 'no_traffic'
    """
    img = Image.open(image_path)

    # Preprocess the image
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract detections for cars
    car_count = 0
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']

    # Count number of cars (assuming 'car' class label is 3, adjust if needed)
    for label in labels:
        if label == 3:  # Assuming 'car' class label is 3, adjust if needed
            car_count += 1

    # Determine category based on car count
    if car_count >= 3:
        category = 'traffic'
    else:
        category = 'no_traffic'

    return category


# Move images into respective traffic/no_traffic folders
for subset in subsets:
    subset_dir = os.path.join(base_dir, subset)

    for folder_name in os.listdir(subset_dir):
        folder_path = os.path.join(subset_dir, folder_name)

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)

                if os.path.isfile(image_path):
                    category = classify_image(image_path)

                    new_path = os.path.join(base_dir, subset, category, file_name)
                    shutil.move(image_path, new_path)
                    print(f'Moved {image_path} to {new_path}')
