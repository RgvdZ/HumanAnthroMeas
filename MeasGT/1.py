"""
1.py
Data Preparation, BMI Extrema Selection, CPD Alignment, and Model Training.
"""

#import os
#import cv2
#import torch
#import shutil
#import pandas as pd
#import numpy as np
#import torch.nn as nn
#import torch.optim as optim
#from torchvision import models, transforms
#from torch.utils.data import Dataset, DataLoader
#from pycpd import AffineRegistration

print("--- TRACE: Script started! ---")

import os
print("--- TRACE: os imported successfully ---")

import cv2
print("--- TRACE: cv2 imported successfully ---")

import shutil
print("--- TRACE: shutil imported successfully ---")

import pandas as pd
print("--- TRACE: pandas imported successfully ---")

import numpy as np
print("--- TRACE: numpy imported successfully ---")

print("--- TRACE: About to import torch... (If it stops here, PyTorch/GPU is the issue) ---")
import torch
print("--- TRACE: torch imported successfully ---")

import torch.nn as nn
print("--- TRACE: torch.nn imported successfully ---")

import torch.optim as optim
print("--- TRACE: torch.optim imported successfully ---")

from torchvision import models, transforms
print("--- TRACE: torchvision imported successfully ---")

from torch.utils.data import Dataset, DataLoader
print("--- TRACE: torch Dataset/DataLoader imported successfully ---")

print("--- TRACE: About to import pycpd... ---")
from pycpd import AffineRegistration
print("--- TRACE: pycpd imported successfully ---")

print("\n--- ALL IMPORTS PASSED! Moving to the rest of the code... ---\n")

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Directories containing your data
SOURCE_DIRS = ['train', 'testA']
# The 5 reference subject IDs (replace with your actual IDs from the CSV)
REF_SUBJECT_IDS = [
    'ciLRduStcoRnAQ-__Qs6QLQWSGY8dxjjK1d0PyYwF-Q', '0llUez60571dPHH-iNUKIhyU8RNcRM3u8cJx27lOkkk', 'zmRG_1E_Nn8ySy4abP899E_O-HTwWNDrt0Me4kyaRm4', 'WnKXOTsT_HOrsCimdLNQ0avrn593cJCE7N3fCWGn9H4', 'N-HpjhR4e4pv_lil_breZsD2Zl6rchnUYp6CN-OId00'
]

# Output directories for the extracted subset
IMG1_DIR = 'img1'
IMGOTHER_DIR = 'imgother'

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224 # Standard size for MobileNetV3

# Measurement columns we want the NN to predict (14 total)
MEASUREMENT_COLS = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height',
    'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

# ==========================================
# 2. DATA PREPARATION & BMI CALCULATIONS
# ==========================================
def load_and_merge_csvs(directories):
    """Reads all 3 CSVs from multiple source directories and merges them."""
    hwg_list, meas_list, map_list = [], [], []

    for d in directories:
        hwg_path = os.path.join(d, 'hwg_metadata.csv')
        meas_path = os.path.join(d, 'measurements.csv')
        map_path = os.path.join(d, 'subject_to_photo_map.csv')

        if os.path.exists(hwg_path): hwg_list.append(pd.read_csv(hwg_path))
        if os.path.exists(meas_path): meas_list.append(pd.read_csv(meas_path))
        if os.path.exists(map_path): map_list.append(pd.read_csv(map_path))

    df_hwg = pd.concat(hwg_list, ignore_index=True).drop_duplicates(subset=['subject_id'])
    df_meas = pd.concat(meas_list, ignore_index=True).drop_duplicates(subset=['subject_id'])
    df_map = pd.concat(map_list, ignore_index=True).drop_duplicates(subset=['subject_id'])

    # Merge on subject_id
    df_merged = pd.merge(df_hwg, df_meas, on='subject_id')
    df_merged = pd.merge(df_merged, df_map, on='subject_id')

    # Calculate BMI = weight_kg / (height_cm / 100)^2
    df_merged['bmi'] = df_merged['weight_kg'] / ((df_merged['height_cm'] / 100) ** 2)
    return df_merged

def create_subset_data():
    """Finds highest/lowest BMI diffs and organizes images into img1 and imgother."""
    print("Preparing data and calculating BMI differences...")
    df = load_and_merge_csvs(SOURCE_DIRS)

    # Create target directories
    for d in [IMG1_DIR, IMGOTHER_DIR]:
        os.makedirs(os.path.join(d, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(d, 'mask_left'), exist_ok=True)

    # Store the 5 Reference Images in img1
    df_refs = df[df['subject_id'].isin(REF_SUBJECT_IDS)].copy()
    df_refs.to_csv(os.path.join(IMG1_DIR, 'refs_metadata.csv'), index=False)

    # Find targets for imgother
    df_pool = df[~df['subject_id'].isin(REF_SUBJECT_IDS)].copy()
    selected_targets = pd.DataFrame()

    # Pairings tracking for the Dataset class later
    training_pairs = []

    for ref_id in REF_SUBJECT_IDS:
        ref_bmi = df_refs[df_refs['subject_id'] == ref_id]['bmi'].values[0]

        # Calculate Delta BMI
        df_pool['bmi_diff'] = df_pool['bmi'] - ref_bmi

        # 30 Highest BMI (Target is much heavier than ref)
        high_bmi_targets = df_pool[df_pool['bmi_diff'] > 0].nlargest(30, 'bmi_diff')

        # 30 Lowest BMI (Target is much lighter than ref)
        # We use nsmallest to get the most negative numbers (largest absolute difference)
        low_bmi_targets = df_pool[df_pool['bmi_diff'] < 0].nsmallest(30, 'bmi_diff')

        combined_targets = pd.concat([high_bmi_targets, low_bmi_targets])
        selected_targets = pd.concat([selected_targets, combined_targets])

        for target_id in combined_targets['subject_id']:
            training_pairs.append({'ref_id': ref_id, 'target_id': target_id})

    selected_targets = selected_targets.drop_duplicates(subset=['subject_id'])
    selected_targets.to_csv(os.path.join(IMGOTHER_DIR, 'targets_metadata.csv'), index=False)
    pd.DataFrame(training_pairs).to_csv('training_pairs.csv', index=False)

    # Copy physical image files
    def copy_masks(df_subset, dest_dir):
        for _, row in df_subset.iterrows():
            sub_id = row['subject_id']
            photo_id = row['photo_id']
            # Search for the photo in source directories
            for src in SOURCE_DIRS:
                front_src = os.path.join(src, 'mask', f"{photo_id}.png")
                side_src = os.path.join(src, 'mask_left', f"{photo_id}.png")

                if os.path.exists(front_src):
                    shutil.copy(front_src, os.path.join(dest_dir, 'mask', f"{photo_id}.png"))
                if os.path.exists(side_src):
                    shutil.copy(side_src, os.path.join(dest_dir, 'mask_left', f"{photo_id}.png"))

    print("Copying mask files to respective folders...")
    copy_masks(df_refs, IMG1_DIR)
    copy_masks(selected_targets, IMGOTHER_DIR)
    print(f"Data Prep Complete. Total target subjects: {len(selected_targets)}. Total Pairs: {len(training_pairs)}")

# ==========================================
# 3. COHERENT POINT DRIFT (CPD) LOGIC
# ==========================================
def extract_contours_as_points(mask_img, max_points=200):
    """Converts a 2D mask into a downsampled point cloud for CPD."""
    # Find external contours of the silhouette
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Flatten contours to a list of (x, y) points
    points = np.vstack(contours).squeeze()
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)

    # Downsample points for CPD speed
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return points

def perform_cpd_alignment(ref_mask, target_mask):
    """
    Performs CPD to align the target_mask towards the ref_mask.
    Returns the warped (aligned) target mask.
    """
    ref_pts = extract_contours_as_points(ref_mask)
    target_pts = extract_contours_as_points(target_mask)

    if ref_pts is None or target_pts is None:
        return target_mask # Fallback if empty mask

    # Run Affine Registration (target -> ref)
    reg = AffineRegistration(**{'X': ref_pts, 'Y': target_pts})
    try:
        # TY contains the aligned points, (B, t) contain the transformation matrices
        TY, (B, t) = reg.register()

        # Create a transformation matrix for cv2.warpAffine
        # Y_aligned = Y * B^T + t -> standard affine
        M = np.zeros((2, 3))
        M[:, :2] = B.T
        M[:, 2] = t

        aligned_target_mask = cv2.warpAffine(target_mask, M, (ref_mask.shape[1], ref_mask.shape[0]))
        return aligned_target_mask
    except Exception as e:
        # If CPD fails to converge, return original target mask
        return target_mask

# ==========================================
# 4. DATASET & DATALOADER
# ==========================================
class VTONMeasurementDataset(Dataset):
    def __init__(self, pairs_csv, refs_csv, targets_csv, transform=None):
        self.pairs = pd.read_csv(pairs_csv)
        self.refs_df = pd.read_csv(refs_csv).set_index('subject_id')
        self.targets_df = pd.read_csv(targets_csv).set_index('subject_id')
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def load_mask(self, directory, photo_id, side=False):
        sub_folder = 'mask_left' if side else 'mask'
        path = os.path.join(directory, sub_folder, f"{photo_id}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback black image if missing
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        ref_id, target_id = pair['ref_id'], pair['target_id']

        ref_photo_id = self.refs_df.loc[ref_id, 'photo_id']
        target_photo_id = self.targets_df.loc[target_id, 'photo_id']

        # Load reference masks (Img1)
        ref_front = self.load_mask(IMG1_DIR, ref_photo_id, side=False)
        ref_side = self.load_mask(IMG1_DIR, ref_photo_id, side=True)

        # Load target masks (ImgOther)
        target_front = self.load_mask(IMGOTHER_DIR, target_photo_id, side=False)
        target_side = self.load_mask(IMGOTHER_DIR, target_photo_id, side=True)

        # Apply Coherent Point Drift Alignment
        aligned_target_front = perform_cpd_alignment(ref_front, target_front)
        aligned_target_side = perform_cpd_alignment(ref_side, target_side)

        # Stack into 4 channels: [ref_front, ref_side, target_front_aligned, target_side_aligned]
        # We normalize to 0-1 range
        stacked_tensor = np.stack([
            ref_front / 255.0,
            ref_side / 255.0,
            aligned_target_front / 255.0,
            aligned_target_side / 255.0
        ], axis=0).astype(np.float32)

        # Extract Ground Truth Measurements for the TARGET
        measurements = self.targets_df.loc[target_id, MEASUREMENT_COLS].values.astype(np.float32)

        return torch.tensor(stacked_tensor), torch.tensor(measurements)

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
def build_model():
    """Modifies MobileNetV3_small to take 4 channels and output 14 measurements."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # 1. Modify the first Conv Layer to accept 4 channels instead of 3
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    # Initialize the new 4th channel weights with the mean of the original 3 channels
    with torch.no_grad():
        model.features[0][0].weight[:, :3] = original_conv.weight
        model.features[0][0].weight[:, 3] = original_conv.weight.mean(dim=1)

    # 2. Modify Classifier to output 14 measurements
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 14)

    return model

# ==========================================
# 6. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Prepare subsets (Uncomment this to run subset creation on first pass)
    create_subset_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 2. Initialize Data
    dataset = VTONMeasurementDataset(
        pairs_csv='training_pairs.csv',
        refs_csv=os.path.join(IMG1_DIR, 'refs_metadata.csv'),
        targets_csv=os.path.join(IMGOTHER_DIR, 'targets_metadata.csv')
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss, Optimizer
    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (images, measurements) in enumerate(dataloader):
            images, measurements = images.to(device), measurements.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, measurements)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print("Training Finished. Saving state dict...")
    torch.save(model.state_dict(), 'vton_mobilenet_weights.pth')
    print("Weights saved successfully to 'vton_mobilenet_weights.pth'")
