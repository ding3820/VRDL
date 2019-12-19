# create new environment
conda create -n env python=3.6
conda activate env

# pytorch installation
conda install pytorch=1.3 torchvision cudatoolkit=10.1 -c pytorch
# opencv
conda install -c menpo opencv3
# fvcore
pip install 'git+https://github.com/facebookresearch/fvcore'
# pycocotools
pip install cython;
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install detectron2
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2

# Prepare the config file.
cp configs/mask_rcnn_R_50_FPN_modified.yaml detectron2/configs/COCO-InstanceSegmentation/
