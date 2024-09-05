import tarfile
import os

# Path to the downloaded tarfile
tarfile_path = "C:/Users/ASUS/Downloads/idd-lite (1).tar.gz"  # Adjust this path as per your file name

# Directory to extract the files into
extract_dir = 'idd_dataset_extracted_final'  # You can change this directory name if you want

try:
    # Open the tarfile
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        # Extract all files
        tar.extractall(extract_dir)

    print("Dataset extracted successfully.")

except tarfile.TarError as e:
    print("Error extracting dataset from tarfile:", e)
except FileNotFoundError:
    print("Tarfile not found. Please check the file path.")
except Exception as e:
    print("An unexpected error occurred:", e)
