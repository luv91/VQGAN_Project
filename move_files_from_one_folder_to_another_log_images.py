import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import cv2



def log_transform(image):
    # convert image to numpy array
    np_img = np.array(image)

    # convert image to HSV
    hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

    # increase saturation, change the second channel which represents saturation
    hsv_img[..., 1] = hsv_img[..., 1] * 1.5  # adjust the multiplier as per needs

    # convert back to RGB
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # apply log transform
    c = 255 / np.log(1 + np.max(rgb_img)) 
    log_transformed = c * (np.log(rgb_img + 1)) 

    # convert back to uint8
    log_transformed = np.array(log_transformed, dtype = np.uint8)

    # create image from numpy array
    return Image.fromarray(log_transformed)

def copy_and_transform_images(data_folder, destination_folder, limit):
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
                # open image and apply log transformation
                img = Image.open(file)
                transformed_img = log_transform(img)

                # save transformed image to destination folder
                transformed_img.save(os.path.join(destination_folder, file.name))

# define paths to your folders
train_data_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/train/'
valid_data_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/valid/'
train_log_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/train_hsv_log_data/'
valid_log_folder = '/Users/luv/Documents/GitHub/VQGAN_Project/flower_data/valid_hsv_log_data/'

# copy and transform images
copy_and_transform_images(train_data_folder, train_log_folder, 50)
copy_and_transform_images(valid_data_folder, valid_log_folder, 50)