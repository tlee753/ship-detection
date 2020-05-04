import sys
import os
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from mrcnn.config import Config
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Config
class AirbusShipDetectionChallengeGPUConfig(Config):
    NAME = 'SHIP-TEST'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50
    SAVE_BEST_ONLY = True
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.05

class InferenceConfig(AirbusShipDetectionChallengeGPUConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='./')
model.load_weights("model-1.h5", by_name=True)

img_path = sys.argv[1]
img_name = os.path.basename(img_path)
img = img_to_array(load_img(img_path))

# Object Detection
result = model.detect([img], verbose=1)

r = result[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            ["ship", "Ship", "ship"], r['scores'], 
                            title="Predictions for " + img_path)

visualize.save_image(img, img_name[:-4] + "-detected", r['rois'], r['masks'], r['class_ids'], r['scores'], ["ship", "Ship", "ship"])

# image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

# AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                                      r["rois"], r["class_ids"],
#                                                      r["scores"], r['masks'])
