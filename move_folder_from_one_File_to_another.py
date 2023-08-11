import os
import shutil
from pathlib import Path

def copy_images_to_one_folder(data_folder, destination_folder, limit):
    # ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # get all directories and sort them
    directories = sorted([d for d in Path(data_folder).iterdir() if d.is_dir()])
    
    # Limit to the first 50 directories
    directories = directories[:limit]

    for directory in directories:
        for file in directory.iterdir():
            # ensure it's a file and not a subdirectory
            if file.is_file():
                shutil.copy2(str(file), destination_folder)

# define paths to your folders
train_data_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/train/'
valid_data_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/valid/'
train_all_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/train_all_only_few/'
valid_all_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/valid_all_only_few/'

# copy images
copy_images_to_one_folder(train_data_folder, train_all_folder, 2)
copy_images_to_one_folder(valid_data_folder, valid_all_folder, 2)
