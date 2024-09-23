import base64
import glob
import io
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
    xyxy2xywh,
)
from utils.torch_utils import select_device, time_synchronized

# Tunable variables
LOG_LEVEL = logging.DEBUG  # Set the logging level
LOG_FILE = "detection_results.log"  # Log file
IMG_FORMATS = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # Supported image formats

# Configure logging (overwrite mode by using 'w' in FileHandler)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Overwrite log file on each run
        logging.StreamHandler(),  # Also log to the console
    ],
)


class YOLOArgs:
    def __init__(self):
        self.weights = "/data/training_code/yolov7/runs/train/v7-e6e/p6/bs96-ep1000-lr_1e-3/weights/best.pt"
        self.img_size = 640  # inference size (pixels)
        self.conf_thres = 0.25  # object confidence threshold
        self.iou_thres = 0.45  # IOU threshold for NMS
        self.device = "0"
        self.augment = False  # augmented inference
        self.classes = None  # filter by class: list, e.g., [0, 1, 2]
        self.agnostic_nms = False  # class-agnostic NMS
        self.save_conf = True  # save confidences in output


def bbu_yolo_detect(
    img_list,  # List of images (PIL or NumPy)
    ops: YOLOArgs,  # YOLO arguments
):
    # Extract parameters from ops (YOLOArgs instance)
    weights = ops.weights
    img_size = ops.img_size
    conf_thres = ops.conf_thres
    iou_thres = ops.iou_thres
    device = ops.device
    augment = ops.augment
    classes = ops.classes
    agnostic_nms = ops.agnostic_nms
    save_conf = ops.save_conf

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != "cpu"

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    if half:
        model.half()

    # Set Dataloader for list of images
    dataset = LoadImagesFromList(img_list, img_size=img_size, stride=stride)

    # Inference warm-up
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, img_size, img_size)
            .to(device)
            .type_as(next(model.parameters()))
        )

    old_img_w = old_img_h = img_size
    old_img_b = 1
    t0 = time.time()
    results = []

    for img_id, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
        )
        t3 = time_synchronized()

        # Process detections
        figure_results = []
        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]

        if len(pred):
            for det in pred:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )
                    result_line = (
                        [int(cls), *xywh, float(conf)]
                        if save_conf
                        else [int(cls), *xywh]
                    )
                    figure_results.append(result_line)

        results.append({"img_id": img_id, "detections": figure_results})
        print(
            f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
        )

    print(f"Total time: {time.time() - t0:.3f}s")
    return results


class LoadImagesFromList:
    def __init__(self, img_list, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.files = img_list  # List of tuples (img_id, image)
        self.nf = len(img_list)
        self.video_flag = [False] * self.nf
        self.mode = "image"
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        img_id, img0 = self.files[self.count]
        self.count += 1

        # Convert PIL image to numpy array (RGB)
        img0 = np.array(img0)

        # Convert RGB to BGR (OpenCV format)
        img0 = img0[:, :, ::-1]

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_id, img, img0, None  # None for cap

    def __len__(self):
        return self.nf


# Collect image names from source directory, matching the logic in LoadImages
def collect_image_names(source):
    p = str(Path(source).absolute())
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.*")))  # collect all files
    elif os.path.isfile(p):
        files = [p]
    else:
        raise Exception(f"ERROR: {p} does not exist")

    images = [
        x for x in files if x.split(".")[-1].lower() in IMG_FORMATS
    ]  # filter image files
    return images  # Return sorted list of image paths


def generate_tmp_dict_images(image_paths):
    tmp_dict = {"images": [], "images_id": []}
    for idx, img_path in enumerate(image_paths):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            tmp_dict["images"].append(img_base64)
            tmp_dict["images_id"].append(idx)  # You can use any unique identifier
    return tmp_dict


def main():
    logging.debug("Starting YOLOv7 detection...")

    # Define the YOLO arguments
    opt = YOLOArgs()

    # Collect image paths from a directory
    image_directory = "/data/dataset/BBU_wind_shield/for_predict_test"
    image_formats = ["jpg", "jpeg", "png", "bmp"]  # Supported image formats
    image_paths = []
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.split(".")[-1].lower() in image_formats:
                image_paths.append(os.path.join(root, file))
    logging.debug(f"Collected image paths: {image_paths}")

    # Generate tmp_dict["images"] and tmp_dict["images_id"]
    tmp_dict = generate_tmp_dict_images(image_paths)

    # Process tmp_dict["images"] to get img_list
    images = tmp_dict["images"]
    img_ids = tmp_dict["images_id"]
    list_len = len(images)
    # Get img_list
    img_list = []
    for i in range(list_len):
        img_data = base64.b64decode(images[i])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_list.append((img_ids[i], image))  # Include img_id

    # Call the detect function
    res = bbu_yolo_detect(
        img_list,
        opt,
    )

    # Process results
    for result in res:
        img_id = result["img_id"]
        detections = result["detections"]
        logging.debug(f"Image ID: {img_id}, Detections: {detections}")


if __name__ == "__main__":
    main()
