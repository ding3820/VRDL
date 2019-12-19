import torch
import torchvision
import detectron2
import cv2
import os
import json
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pycocotools.coco import COCO
from utils import binary_mask_to_rle


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, default="./output/model_final.pth",
                        help="trained model weight path")
    return parser.parse_args()


args = get_arguments()

# Configs
cfg = get_cfg()
cfg.merge_from_file('./detectron2/configs/COCO-InstanceSegmentation/' + 'mask_rcnn_R_50_FPN_modified.yaml')
cfg.MODEL.WEIGHTS = args.model_pth
# cfg.DATASETS.TEST = ("test_dataset",)
# Set the testing threshold for this model.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

coco_test = COCO("datasets/test.json")
coco_dt = []
for imgid in coco_test.imgs:
    image = cv2.imread(
        "datasets/test_images/" + coco_test.loadImgs(ids=imgid)[0]['file_name']
    )[:, :, ::-1]  # load image
    outputs = predictor(image)["instances"]  # run inference of your model
    masks = outputs.pred_masks
    categories = outputs.pred_classes
    scores = outputs.scores
    if len(categories) > 0:  # If any objects are detected in this image
        for i in range(len(scores)):  # Loop all instances
            # save information of the instance in a dictionary then append on
            # coco_dt list
            pred = {}
            # this imgid must be same as the key of test.json
            pred['image_id'] = imgid
            pred['category_id'] = int(categories[i]) + 1
            # save binary mask to RLE, e.g. 512x512 -> rle
            pred['segmentation'] = \
                binary_mask_to_rle(masks[i, :, :].cpu().numpy())
            pred['score'] = float(scores[i])
            coco_dt.append(pred)


# Save the prediction to a json file.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
out_pth = os.path.join(cfg.OUTPUT_DIR, "output.json")
with open(out_pth, "w") as f:
    json.dump(coco_dt, f)
