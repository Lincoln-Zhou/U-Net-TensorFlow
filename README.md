**An implementation of U-Net using TensorFlow framework, with multi-class support.**

## Train on Custom Data
We design the project to find a balance between encapsulation and the ability of code modification, please follow the steps below to train on your custom dataset.
* Annotate your data using tools such as LabelMe
* Use the following methods in `utilities.py` to obtain image and mask data for model input from `json` labels:
  * `labelme_json_to_mask()`
  * `generate_sequence_data()`
  * `data_preprocessing()`
* Follow the pipeline in `train.py`
* Evaluate model performance using `evaluate.py`

**Please don't forget to modify `n_class` in `train.py` and `evaluate.py` according to your needs**.

Please refer to the docstring of aforementioned methods for more info.

## Inference on New Data
*Still working on that part now:)*

## Sample Result
The following results are from tiles of an underground parking lot IPM image. The model was trained for detecting road lanes and markers, please refer to `best.h5`.

The model achieved a `Mean_IoU` score of 86.25% on our current version of dataset.

<img alt="01" height="235" src="https://github.com/Lincoln-Zhou/U-Net-TensorFlow/raw/main/sample_result/01.png" width="600" class="center"/>
<img alt="02" height="235" src="https://github.com/Lincoln-Zhou/U-Net-TensorFlow/raw/main/sample_result/02.png" width="600" class="center"/>
<img alt="03" height="235" src="https://github.com/Lincoln-Zhou/U-Net-TensorFlow/raw/main/sample_result/03.png" width="600" class="center"/>
<img alt="04" height="235" src="https://github.com/Lincoln-Zhou/U-Net-TensorFlow/raw/main/sample_result/04.png" width="600" class="center"/>
<img alt="05" height="235" src="https://github.com/Lincoln-Zhou/U-Net-TensorFlow/raw/main/sample_result/05.png" width="600" class="center"/>


## System Requirements
Software:
* Linux, Windows >= 10, macOS (No CUDA support) >= 10.14 operating systems
* Python packages required in `requirements.txt`
* Python version >= 3.6

Hardware:
* Multicore CPU & CUDA-enabled GPU strongly recommended
* Prefer RAM >= 32 GB, VRAM >= 10 GB
* Prefer SSD for faster data loading

## Acknowledgements
The original U-Net architecture was proposed by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015, please refer to [their paper](https://arxiv.org/abs/1505.04597).

This repo also uses code from [bnsreenu's GitHub repository](https://github.com/bnsreenu/python_for_microscopists).

Sincerely thanks to all the people mentioned above.

## Bug Report
The code hasn't gone through thorough test, and bugs are almost bound to exist, please contact the author or open an issue if you encountered one.

Also, please notice that this repo is primarily developed for research purposes, with no guarantee on model performance, efficiency, etc. Use at your own risk.
