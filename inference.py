import tensorflow as tf
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

import utilities as utl

from typing import List
import os


def infer(images: List[str], model_path: str, split: tuple = None, resize: int = 256, output_reshape: tuple = None):
    """
    Perform semantic segmentation using trained model.
    Notice that the split and resize parameters should be consistent with the training stage.

    :param images: List of image paths
    :param model_path: Path to TensorFlow model file (HDF5 format only)
    :param split: Whether to split image into tiles, pass None to suppress this action, or pass a (N_ROW, N_COL) tuple to split into N_ROW * N_COL tiles.
    :param resize: Whether to resize image to (resize, resize), can drastically decrease RAM/VRAM consumption. Tiles will be resized individually.
    :param output_reshape: Whether to reshape mask to a specific size (h, w). If not, pass None and the final output mask shape will be (n_tile_row * resize, n_tile_col * resize)

    :return: masks sequence in np.ndarray
    """

    assert len(images) > 0, 'Empty input image'
    assert split is None or len(split) == 2, 'Invalid split parameter'
    assert output_reshape is None or len(output_reshape) == 2, 'Invalid reshape parameter'

    try:
        model = tf.keras.models.load_model(model_path)
    except (ImportError, IOError):
        print("Fatal error, please check TensorFlow installation and model file existence, stop.")

        raise RuntimeError

    img_input = []

    for img_path in images:
        img = cv2.imread(img_path, 0)

        if split:
            M, N = split
        else:
            M, N = 1, 1

        img_tiles = utl.split_image(img, M, N)

        imgs = [cv2.resize(x, (resize, resize), interpolation=cv2.INTER_NEAREST) for x in img_tiles]

        img_input.extend(imgs)

    img_input = np.array(img_input)

    img_input = np.expand_dims(img_input, axis=3)
    img_input = tf.keras.utils.normalize(img_input, axis=1)

    with utl.Timer("Predicting..."):
        masks_output = model.predict(img_input)

    masks_argmax = np.argmax(masks_output, axis=3)

    """
    for i in range(32):
        plt.imshow(masks_argmax[i], cmap='jet')
        plt.show()"""

    n_tiles = M * N

    masks_result = []

    for index in range(len(images)):
        current_image_mask_tiles = masks_argmax[n_tiles * index: n_tiles * (index + 1)]

        current_mask = utl.fuse_tiles(current_image_mask_tiles, M, N)

        masks_result.append(current_mask)

    if output_reshape:
        h, w = output_reshape
        masks_result = np.array([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in masks_result])
    else:
        masks_result = np.array(masks_result)

    return masks_result


if __name__ == '__main__':
    images = [f"sample_test/{x}" for x in os.listdir("sample_test")]
    results = infer(images=images, model_path='best.h5', split=(8, 4), resize=256, output_reshape=(5200, 3400))

    for index in range(len(results)):
        mask = results[index]

        plt.imshow(mask, cmap='jet')
        plt.savefig(f"sample_result/{index}.png", dpi=400)
