import os

# Paths
DATA_ROOT = "/path/to/dataset"
MASK_DIR = os.path.join(DATA_ROOT, "masks")          # binary silhouette images (e.g., subject_id.png)
CSV_PATH = os.path.join(DATA_ROOT, "labels.csv")     # columns: subject_id, gender, height_mm, meas_0..meas_13
WARP_DIR = os.path.join(DATA_ROOT, "warps")          # precomputed warp fields
MODEL_SAVE_PATH = "oracle_best.pth"

# Preprocessing
IMAGE_SIZE = 256
NUM_MEASUREMENTS = 14

# CPD
CPD_MAX_ITER = 100
CPD_TOLERANCE = 1e-5
CPD_LAMBDA = 3.0         # regularisation
CPD_BETA = 2.0           # kernel width

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
DEVICE = "cuda"
SEED = 42
