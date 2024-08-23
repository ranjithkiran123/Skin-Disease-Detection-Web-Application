import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Path to your dataset folder
data_folder = r"C:\skin_disease_dataset\images"

# Path to the destination folder for train and test sets
train_folder = r"C:\skin_disease_dataset\train"
test_folder = r"C:\skin_disease_dataset\test"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load metadata CSV file
metadata_csv = r"C:\skin_disease_dataset\HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_csv)

# Split the dataset into train and test sets based on the metadata
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

# Move the train images to the train folder
for _, row in train_metadata.iterrows():
    image_id = row["image_id"]
    class_folder = row["dx"]
    src = os.path.join(data_folder, image_id + ".jpg")
    dst = os.path.join(train_folder, class_folder, image_id + ".jpg")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

# Move the test images to the test folder
for _, row in test_metadata.iterrows():
    image_id = row["image_id"]
    class_folder = row["dx"]
    src = os.path.join(data_folder, image_id + ".jpg")
    dst = os.path.join(test_folder, class_folder, image_id + ".jpg")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
