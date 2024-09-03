import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

# Directories containing the dataset
folder1 = 'dataset/trainingData'
folder2 = 'dataset/testingData'

# Destination directories for train and test sets
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List of all classes (assuming same structure in both folders)
classes = os.listdir(folder1)

# Iterate through each class and combine files from both folders
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Get all files from both folders for the current class
    files = glob(os.path.join(folder1, class_name, '*')) + glob(os.path.join(folder2, class_name, '*'))
    
    # Shuffle and split the dataset (e.g., 80% train, 20% test)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Move the files to the respective directories
    for file in train_files:
        shutil.move(file, os.path.join(train_dir, class_name, os.path.basename(file)))
        
    for file in test_files:
        shutil.move(file, os.path.join(test_dir, class_name, os.path.basename(file)))

print("Data successfully split into train and test sets.")
