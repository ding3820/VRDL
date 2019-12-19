# HW4


# Overview
- Installation
- Training and Inference 
- File description

# Installation
### Requirements
- Linux or macOS
- Python >= 3.6
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install -U 'git+https://github.com/facebookresearch/fvcore'`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC >= 4.9

or simply run `setup.sh`

# Training and Inference 
- Train
    ```shell=
    python train.py
    ```
 - Infernce
    - `--model_pth`: The path to your model weight.
    - Command
        ```shell=
        python test.py --model_pth weight_path
        ```
    - Generate `output.json` under `output/`

# File description
- `train.py`: Train the model.
- `test.py`: Run inference on test data and generate output JSON file.
- `configs/`: Store all the config files. See yacs for more detail.
- `datasets/`:Contain annotation files and train/test data.
- `output/`: The submission JSON files of HW4.
- `setup.sh`: Build the environment.
- `utils.py`: For generating JSON file.
