import sys
import os
import numpy as np
import yaml
from tifffile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops
from skimage.morphology import white_tophat, disk

#STop TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###PARAMETERS###

#paths from Snakefile
INPUT_PATH = sys.argv[1]
OUTPUT_DET_PATH = sys.argv[2]
OUTPUT_SEL_PATH = sys.argv[3]
config_path = sys.argv[4]

with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f) #Dict with all config parameters

p = full_config["algorithms"]["stardist"]["params"]

# Image Pre-processing
CHANNEL_INDEX = p["channel_index"]
NORM_PERCENTILE_MIN = p["norm_percentile_min"]
NORM_PERCENTILE_MAX = p["norm_percentile_max"]
AXIS_NORM = tuple(p["axis_norm"])

# StarDist Model Parameters
STARDIST_MODEL_NAME = p["model_name"]
PROB_THRESHOLD = p["prob_threshold"]
NMS_THRESHOLD = p["nms_threshold"]

# Filter Parameters
TOPHAT_RADIUS = p["tophat_radius"]
MIN_AREA = p["min_area"]
MAX_AREA = p["max_area"]
MAX_MEAN_INTENSITY = p["max_mean_intensity"]
MIN_MEAN_INTENSITY = p["min_mean_intensity"]
MAX_ECCENTRICITY = p["max_eccentricity"]
MIN_SOLIDITY = p["min_solidity"]


###PREPROCESSING###


tiffvideo = imread(INPUT_PATH)

#Matching Papylio's donor-only detection 
tiffvideo = tiffvideo[:, CHANNEL_INDEX, :, :]
mean_video = np.mean(tiffvideo, axis=0)
mean_video = white_tophat(mean_video, footprint=disk(TOPHAT_RADIUS)) #Background correction
img = normalize(mean_video, NORM_PERCENTILE_MIN, NORM_PERCENTILE_MAX, axis=AXIS_NORM) # Normalizaton


###STARDIST DETECTION###


# Load Model
model = StarDist2D.from_pretrained(STARDIST_MODEL_NAME)

# Predict
print("Running StarDist Prediction...")
labels, details = model.predict_instances( #Details is a dictionary with confidences, labels is mask
    img,
    prob_thresh=PROB_THRESHOLD,
    nms_thresh=NMS_THRESHOLD
)

regions = regionprops(labels, intensity_image=img) #Using Scikit regionprops to get properties from StarDist masks

# Extract weighted centroids (y, x)
if len(regions) > 0:
    raw_points = np.array([props.weighted_centroid for props in regions])
else:
    raw_points = np.empty((0, 2))

# remove all edge detections, same as Papylio ( not simulated either)
EDGE_MARGIN = p["edge_margin"]
h, w = img.shape
if len(raw_points) > 0:
    keep = ((raw_points[:, 0] >= EDGE_MARGIN) & (raw_points[:, 0] < h - EDGE_MARGIN) &
            (raw_points[:, 1] >= EDGE_MARGIN) & (raw_points[:, 1] < w - EDGE_MARGIN))
    raw_points = raw_points[keep]
    regions = [r for r, k in zip(regions, keep) if k]

final_detections = raw_points #use the centroids as detections saved to pipeline


###STARDIST SELECTION###


valid_indices = []

for i, props in enumerate(regions): 
    cond_size = MIN_AREA <= props.area <= MAX_AREA 
    cond_intensity1 = props.mean_intensity < MAX_MEAN_INTENSITY
    cond_intensity2 = props.mean_intensity > MIN_MEAN_INTENSITY
    cond_eccentricity = props.eccentricity < MAX_ECCENTRICITY
    cond_solidity = props.solidity > MIN_SOLIDITY

    if cond_size and cond_intensity1 and cond_intensity2 and cond_eccentricity and cond_solidity:
        valid_indices.append(i)

if len(valid_indices) > 0:
    filtered_points = raw_points[valid_indices]
    final_selections = filtered_points
else:
    final_selections = np.empty((0, 2))


###SAVING###


np.save(OUTPUT_DET_PATH, final_detections)
np.save(OUTPUT_SEL_PATH, final_selections)