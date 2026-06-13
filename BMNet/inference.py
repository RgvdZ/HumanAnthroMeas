import torch
import torch.nn as nn
import pandas as pd
import os
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ================= CONFIGURATION =================
WEIGHTS_PATH = "/iitgn/homedirs/preyum.kumar/rgvd/measurement/dataset/codebase/BMNet/checkpoints/bmnet_mnasnet_weights_B.pth"
DATASET_ROOT = "/iitgn/homedirs/preyum.kumar/rgvd/measurement/dataset/Dataset"
SUBJECT_ID = "XAfhSHKVLh1TufsAmAgv7vKa7lmbjgXwP90BMR0za2Y"      # _UIZN2KX9YMinrn-UkUP23ta1UIIA6CSbAKoPgiIMcg is BMI 20.09, 59bc2758681516389ccf4071 train bmi 51
SPLIT_FOLDER = "testB" 
# =================================================

class InferenceDatasetItem:
    """Standalone logic to process a single subject to match model expectations."""
    def __init__(self, dataset_root, split_folder):
        self.img_dir = os.path.join(dataset_root, split_folder)
        self.resize = T.Resize((640, 480))
        self.to_tensor = T.ToTensor()
        self.measurement_columns = [
            'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
            'height', 'hip', 'leg-length', 'shoulder-breadth',
            'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
        ]
        
        self.meas_df = pd.read_csv(os.path.join(self.img_dir, "measurements.csv"))
        self.hwg_df = pd.read_csv(os.path.join(self.img_dir, "hwg_metadata.csv"))
        self.map_df = pd.read_csv(os.path.join(self.img_dir, "subject_to_photo_map.csv"))

    def get_data(self, sub_id):
        photo_ids = self.map_df[self.map_df['subject_id'] == sub_id]['photo_id'].tolist()
        if not photo_ids:
            raise ValueError(f"Subject ID {sub_id} not found in map_df!")
            
        row = self.meas_df[self.meas_df['subject_id'] == sub_id].iloc[0]
        hwg = self.hwg_df[self.hwg_df['subject_id'] == sub_id].iloc[0]

        front_path = os.path.join(self.img_dir, 'mask', f'{photo_ids[0]}.png')
        lateral_id = photo_ids[1] if len(photo_ids) > 1 else photo_ids[0]
        lateral_path = os.path.join(self.img_dir, 'mask_left', f'{lateral_id}.png')

        front_img = Image.open(front_path).convert('L')
        lateral_img = Image.open(lateral_path).convert('L')

        front_tensor = self.to_tensor(self.resize(front_img))
        lateral_tensor = self.to_tensor(self.resize(lateral_img))
        combined_silhouette = torch.cat((front_tensor, lateral_tensor), dim=2)

        norm_height = float(hwg['height_cm']) / 250.0
        norm_weight = float(hwg['weight_kg']) / 200.0

        height_channel = torch.full((1, 640, 960), norm_height, dtype=torch.float32)
        weight_channel = torch.full((1, 640, 960), norm_weight, dtype=torch.float32)

        input_tensor = torch.cat((combined_silhouette, height_channel, weight_channel), dim=0)
        
        gt_values = row[self.measurement_columns].values.astype('float32')
        
        return input_tensor.unsqueeze(0), gt_values

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Model Architecture
    model = models.mnasnet1_0(weights=None)
    # Recreating the exact classifier structure used in your training scripts
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=True),
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 14)
    )
    model = model.to(device)

    # 2. Load Weights with weights_only=False to bypass security restriction
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    # 3. Load Sample and run Inference
    model.eval()
    loader = InferenceDatasetItem(DATASET_ROOT, SPLIT_FOLDER)
    input_tensor, gt_values = loader.get_data(SUBJECT_ID)
    
    with torch.no_grad():
        prediction = model(input_tensor.to(device)).cpu().numpy().flatten()

    # 4. Print Results
    print(f"\nResults for Subject: {SUBJECT_ID}")
    print(f"{'Measurement Name':<20} | {'Prediction':<10} | {'Ground Truth':<10}")
    print("-" * 50)
    
    for i, name in enumerate(loader.measurement_columns):
        print(f"{name:<20} | {prediction[i]:10.2f} | {gt_values[i]:10.2f}")

if __name__ == "__main__":
    main()