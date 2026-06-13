import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR

from dataset import BodyMDataset
from train import fit

import torchvision.models as models
from torchvision.models import MNASNet1_0_Weights


def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 22
    LEARNING_RATE = 1e-3
    TOTAL_EPOCHS = 1024
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "checkpoints/bmnet_mnasnet_weights.pth"

    # --- Absolute Data Paths ---
    DATASET_ROOT = "/Put your ;ocation here"
    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    TEST_A_DIR = os.path.join(DATASET_ROOT, "testA")
    TEST_B_DIR = os.path.join(DATASET_ROOT, "testB")

    # --- Setup Dataset & Splits ---
    torch.manual_seed(42)

    # 1. Load the core train/val source dataset
    train_val_dataset = BodyMDataset(
        mapping_csv=os.path.join(TRAIN_DIR, "subject_to_photo_map.csv"),
        meas_csv=os.path.join(TRAIN_DIR, "measurements.csv"),
        hwg_csv=os.path.join(TRAIN_DIR, "hwg_metadata.csv"),
        img_dir=TRAIN_DIR
    )

    # Calculate exact 90% Train / 10% Validation Split from the train folder
    total_train_val = len(train_val_dataset)
    val_size = int(0.1 * total_train_val)
    train_size = total_train_val - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )

    # 2. Load Test-A and Test-B cleanly from their independent directories
    test_a_dataset = BodyMDataset(
        mapping_csv=os.path.join(TEST_A_DIR, "subject_to_photo_map.csv"),
        meas_csv=os.path.join(TEST_A_DIR, "measurements.csv"),
        hwg_csv=os.path.join(TEST_A_DIR, "hwg_metadata.csv"),
        img_dir=TEST_A_DIR
    )

    test_b_dataset = BodyMDataset(
        mapping_csv=os.path.join(TEST_B_DIR, "subject_to_photo_map.csv"),
        meas_csv=os.path.join(TEST_B_DIR, "measurements.csv"),
        hwg_csv=os.path.join(TEST_B_DIR, "hwg_metadata.csv"),
        img_dir=TEST_B_DIR
    )

    print("=================== DATASET COHORT CONFIGURATION ===================")
    print(f" Source Train Folder Total : {total_train_val}")
    print(f" -> Training Split (90%)   : {train_size}")
    print(f" -> Validation Split (10%) : {val_size}")
    print(f" Independent Test-A Split  : {len(test_a_dataset)}")
    print(f" Independent Test-B Split  : {len(test_b_dataset)}")
    print("====================================================================\n")

    # --- Data Dataloaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_a_loader = DataLoader(test_a_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_b_loader = DataLoader(test_b_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    # --- Model Setup ---
    # Use the updated 'weights' argument instead of the deprecated 'pretrained' flag
    weights = MNASNet1_0_Weights.DEFAULT
    model = models.mnasnet1_0(weights=weights)

    # Your custom regression head remains exactly the same
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=True),
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 14)
    )
    model = model.to(DEVICE)




    # --- Loss and Optimizer ---
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    milestones = [int(0.75 * TOTAL_EPOCHS), int(0.88 * TOTAL_EPOCHS)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # --- Automated Resuming State Verification ---
    start_epoch = 1
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint detected at '{CHECKPOINT_PATH}'. Restoring session states...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        print(f"--> Success! Resuming training execution stream starting at Epoch {start_epoch}")
    else:
        print("No prior checkpoint parameters matched. Constructing parameters from epoch init state.")

    # --- Run Training ---
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_a_loader=test_a_loader,
        test_b_loader=test_b_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        start_epoch=start_epoch,
        total_epochs=TOTAL_EPOCHS,
        measurement_labels=train_val_dataset.measurement_columns,
        scheduler=scheduler
    )

if __name__ == "__main__":
    main()
