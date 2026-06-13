"""
2.py
Inference and evaluation on 10 random samples from 'testb' using the 5-image ensemble.
"""

import os
import cv2
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import models
from pycpd import AffineRegistration

# ==========================================
# CONFIGURATION
# ==========================================
TEST_DIR = 'testB'
IMG1_DIR = 'img1'
IMG_SIZE = 224

MEASUREMENT_COLS = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height',
    'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# HELPER FUNCTIONS (CPD & Model Logic)
# ==========================================
# Keep CPD identical to training phase to maintain feature integrity
def extract_contours_as_points(mask_img, max_points=200):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    points = np.vstack(contours).squeeze()
    if len(points.shape) == 1: points = points.reshape(-1, 2)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return points

def perform_cpd_alignment(ref_mask, target_mask):
    ref_pts = extract_contours_as_points(ref_mask)
    target_pts = extract_contours_as_points(target_mask)
    if ref_pts is None or target_pts is None: return target_mask

    reg = AffineRegistration(**{'X': ref_pts, 'Y': target_pts})
    try:
        TY, (B, t) = reg.register()
        M = np.zeros((2, 3))
        M[:, :2] = B.T
        M[:, 2] = t
        return cv2.warpAffine(target_mask, M, (ref_mask.shape[1], ref_mask.shape[0]))
    except:
        return target_mask

def load_mask(directory, photo_id, side=False):
    sub_folder = 'mask_left' if side else 'mask'
    path = os.path.join(directory, sub_folder, f"{photo_id}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def build_model():
    """Rebuilds the modified 4-channel MobileNetV3 small."""
    model = models.mobilenet_v3_small()
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=4, out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size, stride=original_conv.stride,
        padding=original_conv.padding, bias=False
    )
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 14)
    return model

# ==========================================
# MAIN INFERENCE PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Loading test data...")
    # Read Test B metadata
    map_df = pd.read_csv(os.path.join(TEST_DIR, 'subject_to_photo_map.csv'))
    meas_df = pd.read_csv(os.path.join(TEST_DIR, 'measurements.csv'))
    test_df = pd.merge(map_df, meas_df, on='subject_id')

    # Pick 10 random subjects
    random_samples = test_df.sample(n=10, random_state=42)

    # Read the 5 Reference Imgs metadata
    refs_df = pd.read_csv(os.path.join(IMG1_DIR, 'refs_metadata.csv'))

    # Load Model
    model = build_model().to(device)
    model.load_state_dict(torch.load('vton_mobilenet_weights.pth', map_location=device))
    model.eval()
    print("Model weights loaded successfully.")

    total_error = 0.0

    with torch.no_grad():
        for i, row in random_samples.iterrows():
            target_photo_id = row['photo_id']
            ground_truth = row[MEASUREMENT_COLS].values.astype(np.float32)

            target_front = load_mask(TEST_DIR, target_photo_id, side=False)
            target_side = load_mask(TEST_DIR, target_photo_id, side=True)

            # The NN runs 5 times (once against each reference image)
            ensemble_predictions = []

            for _, ref_row in refs_df.iterrows():
                ref_photo_id = ref_row['photo_id']

                ref_front = load_mask(IMG1_DIR, ref_photo_id, side=False)
                ref_side = load_mask(IMG1_DIR, ref_photo_id, side=True)

                # Align test masks to the specific reference mask
                aligned_front = perform_cpd_alignment(ref_front, target_front)
                aligned_side = perform_cpd_alignment(ref_side, target_side)

                # Stack to 4 channels
                stacked_tensor = np.stack([
                    ref_front / 255.0, ref_side / 255.0,
                    aligned_front / 255.0, aligned_side / 255.0
                ], axis=0).astype(np.float32)

                input_tensor = torch.tensor(stacked_tensor).unsqueeze(0).to(device) # Add batch dim

                # Predict
                pred = model(input_tensor).cpu().numpy().squeeze()
                ensemble_predictions.append(pred)

            # Average the 5 predictions to create a robust final answer
            final_prediction = np.mean(ensemble_predictions, axis=0)

            # Calculate error
            sample_error = np.mean(np.abs(final_prediction - ground_truth))
            total_error += sample_error

            print(f"\n--- Subject ID: {row['subject_id']} ---")
            print("Predicted vs Ground Truth:")
            for j, col_name in enumerate(MEASUREMENT_COLS):
                print(f"  {col_name.ljust(20)}: Pred = {final_prediction[j]:.2f} | GT = {ground_truth[j]:.2f}")
            print(f"Mean Absolute Error for this subject: {sample_error:.2f} cm")

    print(f"\nOverall Mean Absolute Error across 10 random samples: {total_error/10:.2f} cm")
