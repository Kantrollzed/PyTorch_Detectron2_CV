import os
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

def visualize_predictions(frame, metadata, preds):
    v = Visualizer(frame[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE
                   )
    v = v.draw_instance_predictions(preds["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]

if __name__ == "__main__":
    models_pretrained_list = [
        "faster_rcnn_R_50_FPN_3x_700_box.pth",
        "mask_rcnn_R_50_C4_3x_700_box.pth",
        "mask_rcnn_R_50_FPN_3x_700_box.pth"
    ]
    models_pretrained_type_list = [
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ]
    model_path = models_pretrained_list[2]
    MetadataCatalog.get("metadata_classes").set(thing_classes=["box"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(models_pretrained_type_list[2]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80

    predictor = DefaultPredictor(cfg)
    metadata_classes = MetadataCatalog.get("metadata_classes")

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('result/rezult.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))

    frame_det = 2
    ret, frame = cap.read()
    preds = predictor(frame)
    del ret, frame

    while cap.isOpened():
        ret, frame = cap.read()

        if (frame_det == 2):
            preds = predictor(frame)
            frame_det = 0
        frame_det += 1

        image = visualize_predictions(frame, metadata_classes, preds)
        out.write(image)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
