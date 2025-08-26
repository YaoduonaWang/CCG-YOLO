# Function to read and concatenate multispectral images into single Numpy array

import cv2
import numpy as np

def read_multispectral(
    image_name,
    folder_path,
    channel_types,
) -> np.ndarray:
    
    """
    Returns combined numpy array for prediction. 
    Important - images should be loaded in the same order as trained
    Currently, the images for N channels are loaded for training as follows:
    images_ch1, ..., images_chN, images_rgb
    Channel types should follow the same order:
    {"ch1.tif", ..., "chN.tif", "rgb.tif"}
    """

    image_list = []

    for k in channel_types:
        full_path = f"{folder_path}/{image_name}{k}"
        

        if "rgb" not in k:
            band = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            band = np.expand_dims(band, axis=-1)
        else:
            band = cv2.imread(full_path)

        image_list.append(band)
    
    image = np.concatenate(image_list, axis = -1)

    return image

def extract_rgb(
    multispectral_array
) -> np.ndarray:
    
    bgr_image = multispectral_array[:, :, -3:]  # This gets the last 3 channels in BGR order
    rgb_image = bgr_image[:, :, ::-1]

    return rgb_image

def extract_bgr(
    multispectral_array
) -> np.ndarray:
    
    bgr_image = multispectral_array[:, :, -3:]  # This gets the last 3 channels in BGR order

    return bgr_image
