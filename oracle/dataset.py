import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from cpd_warp import compute_warp_cpd

def load_metadata():
    """Load CSV and return dataframe."""
    df = pd.read_csv(CSV_PATH)
    # expected columns: subject_id, gender, height_mm, meas_0 ... meas_13
    meas_cols = [f"meas_{i}" for i in range(NUM_MEASUREMENTS)]
    return df, meas_cols

def compute_height_scale_factor(mask, height_mm):
    """Compute mm/pixel scale from silhouette height in pixels."""
    # mask is binary; find bounding box height
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return 1.0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    pixel_height = ymax - ymin + 1
    if pixel_height <= 0:
        return 1.0
    return height_mm / pixel_height

class OracleDataset(Dataset):
    """
    Dataset that returns (mask_A, warp_field_mm, delta_y) for training.
    Warp fields are precomputed and saved as .npy files in WARP_DIR.
    """
    def __init__(self, pairs, df, meas_cols, warp_dir=WARP_DIR, augment=True):
        self.pairs = pairs          # list of (idx_A, idx_B) into df
        self.df = df.reset_index(drop=True)
        self.meas_cols = meas_cols
        self.warp_dir = warp_dir
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        iA, iB = self.pairs[idx]
        rowA = self.df.iloc[iA]
        rowB = self.df.iloc[iB]

        # Load masks
        mask_A = cv2.imread(os.path.join(MASK_DIR, f"{rowA['subject_id']}.png"), 0)
        mask_B = cv2.imread(os.path.join(MASK_DIR, f"{rowB['subject_id']}.png"), 0)
        mask_A = cv2.resize(mask_A, (IMAGE_SIZE, IMAGE_SIZE))
        mask_B = cv2.resize(mask_B, (IMAGE_SIZE, IMAGE_SIZE))
        mask_A = (mask_A > 127).astype(np.float32)
        mask_B = (mask_B > 127).astype(np.float32)

        # Load precomputed warp field (pixel units), scaled to mm later
        warp_path = os.path.join(self.warp_dir, f"{rowA['subject_id']}_to_{rowB['subject_id']}.npy")
        if os.path.exists(warp_path):
            warp_pixel = np.load(warp_path)  # (H,W,2)
        else:
            # Compute on the fly (not recommended for training speed)
            warp_pixel = compute_warp_cpd(mask_A, mask_B)
            # save for later use
            os.makedirs(self.warp_dir, exist_ok=True)
            np.save(warp_path, warp_pixel)

        # Scale warp to mm using height of A
        scale_A = compute_height_scale_factor(mask_A, rowA['height_mm'])
        warp_mm = warp_pixel * scale_A

        # Ground truth delta
        y_A = rowA[self.meas_cols].values.astype(np.float32)
        y_B = rowB[self.meas_cols].values.astype(np.float32)
        delta_y = y_B - y_A

        # Augmentation (if enabled)
        if self.augment:
            # Random horizontal flip (p=0.5)
            if np.random.rand() > 0.5:
                mask_A = np.fliplr(mask_A)
                warp_mm = np.fliplr(warp_mm)
                warp_mm[:, :, 0] = -warp_mm[:, :, 0]   # mirror x-component
                # For measurements that are asymmetric (e.g., left/right), you may need to adjust signs;
                # assuming all measurements are symmetric or we don't flip them in output delta.
                # Here we do not flip delta_y, as it's the difference of symmetric measurements.
            # small affine jitter
            # (implementation left as exercise: random rotation/scale/translation)

        # Normalise input warp field to roughly [-1, 1]
        warp_mm_norm = warp_mm / 500.0  # 500 mm typical max displacement

        # Build input tensor: (3, H, W)
        input_tensor = np.concatenate([mask_A[np.newaxis], warp_mm_norm.transpose(2,0,1)], axis=0)  # (3,H,W)

        return torch.FloatTensor(input_tensor), torch.FloatTensor(delta_y)

def create_stratified_pairs(df, num_pairs=10000, seed=SEED):
    """
    Generate pairs (A, B) ensuring coverage of all measurement deltas.
    Strategy: sample 'hard' pairs across BMI groups and 'easy' same-group pairs.
    """
    np.random.seed(seed)
    subjects = df.index.tolist()
    # define BMI groups based on dataset columns (assuming BMI is present or we can derive from height/weight)
    # For demonstration, we'll use the BMI columns given in distribution table.
    # Let's assume the CSV also has 'bmi_group' column.
    bmi_groups = df['bmi_group'].values  # e.g., 'BMI<18.5', 'BMI18.5-25', ...
    genders = df['gender'].values

    pairs = []
    group_indices = {}
    for group in np.unique(bmi_groups):
        group_indices[group] = df.index[df['bmi_group'] == group].tolist()

    # Generate cross-group pairs (hard) and within-group (easy) with ratio 0.3 hard, 0.7 easy
    n_hard = int(0.3 * num_pairs)
    n_easy = num_pairs - n_hard

    # Hard pairs: pick A from one group, B from another far group
    groups_list = list(group_indices.keys())
    for _ in range(n_hard):
        gA = np.random.choice(groups_list)
        gB = np.random.choice([g for g in groups_list if g != gA])
        if group_indices[gA] and group_indices[gB]:
            a = np.random.choice(group_indices[gA])
            b = np.random.choice(group_indices[gB])
            pairs.append((a, b))

    # Easy pairs: same group
    for _ in range(n_easy):
        g = np.random.choice(groups_list)
        if len(group_indices[g]) >= 2:
            a, b = np.random.choice(group_indices[g], 2, replace=False)
            pairs.append((a, b))

    np.random.shuffle(pairs)
    return pairs[:num_pairs]

def precompute_all_warps(pairs, df, warp_dir=WARP_DIR):
    """Generate and save all warp fields for the given pairs."""
    os.makedirs(warp_dir, exist_ok=True)
    for iA, iB in pairs:
        rowA = df.iloc[iA]
        rowB = df.iloc[iB]
        mask_A = cv2.imread(os.path.join(MASK_DIR, f"{rowA['subject_id']}.png"), 0)
        mask_B = cv2.imread(os.path.join(MASK_DIR, f"{rowB['subject_id']}.png"), 0)
        mask_A = cv2.resize(mask_A, (IMAGE_SIZE, IMAGE_SIZE))
        mask_B = cv2.resize(mask_B, (IMAGE_SIZE, IMAGE_SIZE))
        mask_A = (mask_A > 127).astype(np.uint8)
        mask_B = (mask_B > 127).astype(np.uint8)

        out_path = os.path.join(warp_dir, f"{rowA['subject_id']}_to_{rowB['subject_id']}.npy")
        if not os.path.exists(out_path):
            warp = compute_warp_cpd(mask_A, mask_B)
            np.save(out_path, warp)
        # also precompute the reverse direction
        out_path_rev = os.path.join(warp_dir, f"{rowB['subject_id']}_to_{rowA['subject_id']}.npy")
        if not os.path.exists(out_path_rev):
            warp_rev = compute_warp_cpd(mask_B, mask_A)
            np.save(out_path_rev, warp_rev)
