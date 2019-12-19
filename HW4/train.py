import os
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

# Register the dataset
setup_logger()
register_coco_instances(
    "train_dataset", {}, "datasets/pascal_train.json", "datasets/train_images")
register_coco_instances(
    "test_dataset", {}, "datasets/test.json", "datasets/test_images")

# Configs
cfg = get_cfg()
cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/" + "mask_rcnn_R_50_FPN_modified.yaml")
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
