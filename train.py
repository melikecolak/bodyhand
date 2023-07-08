
# -*- coding: utf-8 -*-
"""
Created on Tue May 1 00:28:15 2023

@author: melike.colak
"""
"This development was implemented from the paper "Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild"."

import torch
torch.__version__
import torchvision
torchvision.__version__

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import os
import xml.etree.ElementTree as ET
import numpy as np
from fvcore.common.file_io import PathManager
from typing import List, Tuple, Union
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import random
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import pandas as pd
__all__ = ["load_voc_instances", "register_pascal_voc"]
CLASS_NAMES = ("hand", "body",)
#%%
import os
import pandas as pd

jpeg_dir = './BodyHands/Data/VOC2007/JPEGImages'
anno_dir = './BodyHands/Data/VOC2007/Annotations'

jpeg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]

df = pd.DataFrame(jpeg_files, columns=['jpeg_file'])
df['annotation_exists'] = 0

for i, row in df.iterrows():
    base_name = os.path.splitext(row['jpeg_file'])[0]
    
    anno_file = os.path.join(anno_dir, base_name + '.xml')
    
    if os.path.exists(anno_file):
        df.loc[i, 'annotation_exists'] = 1

print(df)

#%%
def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
 
        hand_annotations = {}
        body_annotations = {}
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls == "hand":
                hand_px = [float(bndbox.find(x).text) for x in ["x1", "x2", "x3", "x4"]]
                hand_py = [float(bndbox.find(x).text) for x in ["y1", "y2", "y3", "y4"]]
                hand_poly = [(x, y) for x, y in zip(hand_px, hand_py)]
                if len(hand_poly) < 3:
                    print(f"Warning: hand_poly has less than 3 points: {hand_poly}")
            else:
                body_px = [bbox[0], bbox[2], bbox[2], bbox[0]]
                body_py = [bbox[1], bbox[1], bbox[3], bbox[3]]
                body_poly = [(x, y) for x, y in zip(body_px, body_py)]
                if len(body_poly) < 3:
                    print(f"Warning: body_poly has less than 3 points: {body_poly}")
            body_id = int(obj.find("body_id").text)
            if cls == "hand":
                hand_ann = {
                        "category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                        "body_id": body_id, "segmentation": [hand_poly],
                    }
                if body_id in hand_annotations:
                    pass
                else:
                    hand_annotations[body_id] = []
                hand_annotations[body_id].append(hand_ann)
            else:
                body_ann = {
                     "category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                     "body_id": body_id, "segmentation": [body_poly], 
                    }
                if body_id in body_annotations:
                    pass 
                else:
                    body_annotations[body_id] = []
                body_annotations[body_id].append(body_ann)

        instances = []
        for body_id in hand_annotations:
            body_ann = body_annotations[body_id][0]
            for hand_ann in hand_annotations[body_id]:
                hand_ann["body_box"] = body_ann["bbox"]
                instances.append(hand_ann)
            body_ann["body_box"] = body_ann["bbox"]
            instances.append(body_ann)

        r["annotations"] = instances
        dicts.append(r)

    return dicts

def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names ))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )

splits = ["train", "test"]
dirname = "./BodyHands/Data/VOC2007/"
for split in splits:
    register_pascal_voc("BodyHands_" + split , dirname, split, 2007)
#%%
dataset_dicts = DatasetCatalog.get("BodyHands_train")
metadata = MetadataCatalog.get("BodyHands_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('image', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)  

#%%

"------Training stage------"

cfg = get_cfg()
cfg.MODEL.DEVICE= 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("BodyHands_train",)  
cfg.DATASETS.TEST = ("BodyHands_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 1000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # I have to choose 128 due to lack of memory. (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
#%%

"------ Plotting Accuracy Values ------- "

metrics_df = pd.read_json("MAX_ITER=1000/metrics.json", orient="records", lines=True)
mdf = metrics_df.sort_values("iteration")
fig, ax = plt.subplots()
mdf1 = mdf[~mdf["mask_rcnn/accuracy "].isna()]
ax.plot(mdf1["iteration"], mdf1["mask_rcnn/accuracy "], c="C0", label="train")
if "mask_rcnn/accuracy " in mdf.columns:
    mdf2 = mdf[~mdf["mask_rcnn/accuracy "].isna()]
    ax.plot(mdf2["iteration"], mdf2["mask_rcnn/accuracy "], c="C1", label="validation")
# ax.set_ylim([0, 0.5])
ax.legend()
ax.set_title("Accuracy curve")
plt.show()

#%% 

"------Evaluation stage------"


cfg.MODEL.WEIGHTS = os.path.join("MAX_ITER=1000", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("BodyHands_test", )

predictor = DefaultPredictor(cfg)
test_dataset = build_detection_test_loader(cfg, "BodyHands_test")
evaluator = COCOEvaluator("BodyHands_test", cfg, False, output_dir="./evalutate/")
metrics =inference_on_dataset(predictor.model, test_dataset, evaluator)

print(metrics["bbox"]["AP"])  # to print bounding box mAP score

" bbox AP : 81.22  "

#%%


" -------- Save the predicted images with class and segmentation information------------" 

os.makedirs("predicts_1000", exist_ok=True)

dataset_dicts = load_voc_instances("./BodyHands/Data/VOC2007/", "test", CLASS_NAMES)
for d in random.sample(dataset_dicts, 20):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("BodyHands_test"), 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(f"predicts_1000/{os.path.basename(d['file_name'])}", v.get_image()[:, :, ::-1])

