import os
import kaggle

# Set the folder path
folder_path = "data/raw"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.chmod(".", 0o777)
    os.makedirs(folder_path)

# Download the dataset using the Kaggle API
kaggle.api.dataset_download_files('aravindhannamalai/dl-dataset', path=folder_path, unzip=True)
