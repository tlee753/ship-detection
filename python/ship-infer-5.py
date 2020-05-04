import sys
import os
import pandas as pd
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from mrcnn.config import Config
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Mask parsing
mask = pd.read_csv("./masks.csv")
print(mask.head(5))

# ground truth function
def load_image_gt(image, mask, config):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    # image = dataset.load_image(image_id)
    # mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    return class_ids, bbox, mask

# Config
class AirbusShipDetectionChallengeGPUConfig(Config):
    NAME = 'ship-test-'
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

gt_class_id, gt_bbox, gt_mask = load_image_gt(img, mask, inference_config)

# AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                                      r["rois"], r["class_ids"],
#                                                      r["scores"], r['masks'])

