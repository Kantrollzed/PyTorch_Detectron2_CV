import os
import numpy as np
import json
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.utils.visualizer import ColorMode


def get_dicts(img_dir, name_json):
    name_json_file = name_json
    json_file = os.path.join(img_dir, name_json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == "__main__":
    
    dataset_name  = "Dataset_name"
    class_list = ["box"]
    models_pretrained_type_list = [
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ]
    model_type_title = models_pretrained_type_list[2]

    for d in ["train", "val"]:
        DatasetCatalog.register(dataset_name + "_" + d, lambda d = d: get_dicts(dataset_name + "/" + d, "via_region_data.json"))
        MetadataCatalog.get(dataset_name + "_" + d).set(thing_classes = class_list)
    train_metadata = MetadataCatalog.get(dataset_name + "_train")

    # Visualize connecting with dataset
    dataset_dicts = get_dicts(dataset_name + "/train", "via_region_data.json")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        print(vis.get_image().shape)
        cv2.imshow("image", vis.get_image()[:, :, ::-1])
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    # Set Configurations
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_type_title))
    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_type_title)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 700
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    # Testing model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = (dataset_name + "_val",)
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(d["file_name"])
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],
                   metadata=train_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", v.get_image()[:, :, ::-1])

    cv2.waitKey()
    cv2.destroyAllWindows()