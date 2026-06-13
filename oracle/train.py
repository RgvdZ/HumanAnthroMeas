import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from config import *
from dataset import OracleDataset, create_stratified_pairs, precompute_all_warps, load_metadata
from model import Oracle
from utils import compute_measurement_stds, WeightedSmoothL1Loss
import os

def main():
    # 1. Load metadata
    df, meas_cols = load_metadata()
    # Assume the CSV also contains 'bmi_group' column; if not, create it from BMI formula.
    # For demonstration, we'll just map the given distribution categories.

    # 2. Create train pairs
    train_pairs = create_stratified_pairs(df, num_pairs=20000)  # adjust size
    # Precompute warp fields for all training pairs (offline, one time)
    precompute_all_warps(train_pairs, df, WARP_DIR)

    # 3. Create datasets
    train_dataset = OracleDataset(train_pairs, df, meas_cols, WARP_DIR, augment=True)
    # Optionally, split val set from same df (by subject ids, not pairs)

    # Compute measurement stds for weighting
    stds = compute_measurement_stds(train_dataset, NUM_MEASUREMENTS)
    print("Measurement stds:", stds.numpy())

    # 4. Model, loss, optimizer
    model = Oracle().to(DEVICE)
    criterion = WeightedSmoothL1Loss(stds)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved.")

    print("Training complete. Best loss:", best_loss)

if __name__ == "__main__":
    main()
