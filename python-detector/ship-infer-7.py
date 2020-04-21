import os
import time
import sys
import errno
import json
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
import numpy as np
import pandas as pd
from skimage.morphology import label
from skimage.io import imread

TRAIN_DATA_PATH = "./input"
SHIP_CLASS_NAME = "ship"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)

masks = pd.read_csv("./masks.csv")
masks.head()

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]

def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=SHAPE):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# ================
# Unique Image Ids
# ================

masks['ships'] = masks['EncodedPixels'].map(lambda encoded_pixels: 1 if isinstance(encoded_pixels, str) else 0)
# print(masks.head())

unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'})
unique_img_ids['RleMaskList'] = masks.groupby('ImageId')['EncodedPixels'].apply(list)
unique_img_ids = unique_img_ids.reset_index()
unique_img_ids = unique_img_ids[unique_img_ids['ships'] > 0]
# print(unique_img_ids.head())

# =============
# Dataset Class
# =============

class AirbusShipDetectionChallengeDataset(utils.Dataset):
    def __init__(self, image_file_dir, ids, masks, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
        super().__init__(self)
        self.image_file_dir = image_file_dir
        self.ids = ids
        self.masks = masks
        self.image_width = image_width
        self.image_height = image_height

        self.add_class(SHIP_CLASS_NAME, 1, SHIP_CLASS_NAME)
        self.load_dataset()

    def load_dataset(self):
        for index, row in self.ids.iterrows():
            image_id = row['ImageId']
            image_path = os.path.join(self.image_file_dir, image_id)
            rle_mask_list = row['RleMaskList']
            self.add_image(
                SHIP_CLASS_NAME,
                image_id=image_id,
                path=image_path,
                width=self.image_width, height=self.image_height,
                rle_mask_list=rle_mask_list)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        rle_mask_list = info['rle_mask_list']
        mask_count = len(rle_mask_list)
        mask = np.zeros([info['height'], info['width'], mask_count],
                        dtype=np.uint8)
        i = 0
        for rel in rle_mask_list:
            if isinstance(rel, str):
                np.copyto(mask[:, :, i], rle_decode(rel))
            i += 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == SHIP_CLASS_NAME:
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)

# ========================
# Prepare and Load Dataset
# ========================

dataset_val = AirbusShipDetectionChallengeDataset(
    image_file_dir=TRAIN_DATA_PATH,
    ids=unique_img_ids,
    masks=masks)
dataset_val.prepare()

# ============
# Config Class
# ============

class AirbusShipDetectionChallengeGPUConfig(Config):
    NAME = 'SHIP-TEST'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50
    SAVE_BEST_ONLY = True
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.05

config = AirbusShipDetectionChallengeGPUConfig()

# ==================
# Interference Setup
# ==================

class InferenceConfig(AirbusShipDetectionChallengeGPUConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
infer_model = modellib.MaskRCNN(mode="inference",
                                config=inference_config,
                                model_dir="./")

# ==========
# Load Model
# ==========

model_path = "./model-1.h5"
# print("Loading weights from ", model_path)
infer_model.load_weights(model_path, by_name=True)

# ===========
# Input Files
# ===========

image_files = sys.argv[1:]
# print(image_files)

def getImageIdsFromFileNames(image_files):
    image_ids = []

    for image_file in image_files:
        image_file = os.path.basename(image_file)
        row = unique_img_ids.loc[unique_img_ids["ImageId"] == image_file]
        image_ids.append(row.index[0])

    return image_ids

image_ids = getImageIdsFromFileNames(image_files)
# print("image ids", image_ids)

# ============
# Interference
# ============

APs = []
metrics = dict()
for image_id in image_ids:
    image_name = os.path.basename(dataset_val.image_reference(image_id))
    print("===============================")
    print("Image reference: ", image_name)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    results = infer_model.detect([image], verbose=1)
    r = results[0]
    visualize.save_image(image, image_name[:-4] + "-detected", r['rois'], r['masks'], r['class_ids'], r['scores'], ["ship", "Ship", "ship"])
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
    
    APs.append(AP)

    image_metrics = dict()
    image_metrics["AP"] = AP
    image_metrics["precisions"] = precisions.tolist()
    image_metrics["recalls"] = recalls.tolist()
    image_metrics["overlaps"] = overlaps.tolist()

    # print(json.dumps(image_metrics))
    metrics[image_name] = image_metrics

print("===============================")

# ==============
# Metrics Output
# ==============

# print("mAP: ", np.mean(APs))
with open("./output/metrics.json", "w+") as json_file:
    json.dump(metrics, json_file, indent=1)
