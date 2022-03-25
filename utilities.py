import numpy as np
from cv2 import cv2
from tensorflow.keras.utils import normalize
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import os


def split_image(image: np.ndarray, M: int = 2, N: int = 2):
    M, N = image.shape[0] // M, image.shape[1] // N

    tiles = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]

    return tiles


def labelme_json_to_mask(path: str, output: str):
    """
    Convert labelme json label data to image and mask image.
    Warning: This operation can be slow on large dataset

    :param path: Directory containing labelme json data, ideally with no other files present.
    :param output: Output directory.
    :return: None
    """
    jsons = os.listdir(path)

    for json in tqdm(jsons):
        if not json.endswith('.json'):
            continue

        name = json[:-5]

        output_path = os.path.join(output, name)
        os.mkdir(output_path)
        os.system("labelme_json_to_dataset " + os.path.join(path, json) + " -o " + output_path)


def generate_sequence_data(path: str, split: tuple = None, resize: int = 256):
    """
    Generate sequential image/mask data for TensorFlow processing

    :param path: Path to directory containing image and mask data, should be the output location of labelme_json_to_mask() method, please refer to README.md for directory structure example.
    Structure:

    directory_name|

    -image1|
        -img.png -label.png
    -image2|
        -img.png -label.png
    ...

    :param split: Whether to split image into tiles, pass None to suppress this action, or pass a (N_ROW, N_COL) tuple to split into N_ROW * N_COL tiles.
    :param resize: Whether to resize image to (resize, resize), can drastically decrease RAM/VRAM consumption. Tiles will be resized individually.

    :return: train_images and train_masks data, in np.ndarray
    """

    assert split is None or len(split) == 2, 'Invalid split parameter'

    train_images, train_masks = [], []

    for index in tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path, str(index), 'img.png'), 0)
        mask = cv2.imread(os.path.join(path, str(index), 'label.png'), 0)

        if split:
            M, N = split

            img_tiles = split_image(img, M, M)
            mask_tiles = split_image(mask, M, N)
        else:
            img_tiles, mask_tiles = [img], [mask]

        imgs = [cv2.resize(img, (resize, resize), interpolation=cv2.INTER_NEAREST) for img in img_tiles]
        masks = [cv2.resize(mask, (resize, resize), interpolation=cv2.INTER_NEAREST) for mask in mask_tiles]

        train_images.extend(imgs)
        train_masks.extend(masks)

    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    return train_images, train_masks


def data_preprocessing(images: np.ndarray, masks: np.ndarray):
    """
    Perform basic preprocessing steps including label encoding, dim expanding and normalization.

    :param images: Sequential image data, ideally output from generate_sequence_data()
    :param masks: Sequential mask data
    :return: processed image and input mask data, in np.ndarray
    """

    labelencoder = LabelEncoder()
    n, h, w = masks.shape

    train_masks_reshaped = masks.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    train_images = np.expand_dims(images, axis=3)
    train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    return train_images, train_masks_input
