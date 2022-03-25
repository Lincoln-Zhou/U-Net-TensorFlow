import numpy as np
from tqdm import tqdm

import os


def split_image(image: np.ndarray, M: int = 2, N: int = 2):
    M, N = image.shape[0] // M, image.shape[1] // N

    tiles = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]

    return tiles


def labelme_json_to_mask(path: str, output: str):
    jsons = os.listdir(path)

    for json in tqdm(jsons):
        if not json.endswith('.json'):
            continue

        name = json[:-5]

        output_path = os.path.join(output, name)
        os.mkdir(output_path)
        os.system("labelme_json_to_dataset " + os.path.join(path, json) + " -o " + output_path)
