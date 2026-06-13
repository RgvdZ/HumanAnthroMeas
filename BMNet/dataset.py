import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class BodyMDataset(Dataset):
    def __init__(self, mapping_csv, meas_csv, hwg_csv, img_dir, transform=None):
        map_df = pd.read_csv(mapping_csv)
        meas_df = pd.read_csv(meas_csv)
        hwg_df = pd.read_csv(hwg_csv)

        photos_df = map_df.groupby('subject_id')['photo_id'].apply(list).reset_index()
        self.data_df = photos_df.merge(meas_df, on='subject_id').merge(hwg_df, on='subject_id')
        self.img_dir = img_dir

        self.resize = T.Resize((640, 480))
        self.to_tensor = T.ToTensor()

        self.measurement_columns = [
            'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
            'height', 'hip', 'leg-length', 'shoulder-breadth',
            'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
        ]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # Keep track of the real subject string identifier
        sub_id = str(row['subject_id'])

        photo_ids = row['photo_id']
        front_photo_id = photo_ids[0]
        lateral_photo_id = photo_ids[1] if len(photo_ids) > 1 else photo_ids[0]

        front_path = os.path.join(self.img_dir, 'mask', f'{front_photo_id}.png')
        lateral_path = os.path.join(self.img_dir, 'mask_left', f'{lateral_photo_id}.png')

        # -------------------------------------------------------------------------------------
        # [MODIFIED] CRITICAL BUG FIX: Re-added .convert('L')
        # Without explicit grayscale conversion, PIL loads binary/silhouette png images as
        # RGB (3 channels) or Palette-indexed modes. T.ToTensor() would then produce a
        # [3, 640, 480] tensor instead of [1, 640, 480], breaking the spatial dimensions
        # and matrix concatenations down the line.
        # -------------------------------------------------------------------------------------
        front_img = Image.open(front_path).convert('L')
        lateral_img = Image.open(lateral_path).convert('L')

        front_tensor = self.to_tensor(self.resize(front_img))
        lateral_tensor = self.to_tensor(self.resize(lateral_img))

        combined_silhouette = torch.cat((front_tensor, lateral_tensor), dim=2)

        # -------------------------------------------------------------------------------------
        # [MODIFIED] ANNOUNCEMENT ON NORMALIZATION QUESTION ("should we do it?"):
        # Yes, you absolutely must do this! Neural networks struggle when raw numbers like
        # 180.0 (height) and 85.0 (weight) are injected straight into spatial feature maps
        # alongside 0.0-1.0 normalized pixel tensors. Min-max scaling these static metadata
        # features down to a 0.0 to 1.0 bounding range keeps your gradient updates stable and
        # prevents height/weight channels from blowing up or overpowering the silhouette features.
        # -------------------------------------------------------------------------------------
        raw_height = float(row['height_cm'])
        raw_weight = float(row['weight_kg'])
        norm_height = raw_height / 250.0
        norm_weight = raw_weight / 200.0

        height_channel = torch.full((1, 640, 960), norm_height, dtype=torch.float32)
        weight_channel = torch.full((1, 640, 960), norm_weight, dtype=torch.float32)

        final_input = torch.cat((combined_silhouette, height_channel, weight_channel), dim=0)

        measurements = row[self.measurement_columns].values.astype('float32')
        measurements_tensor = torch.tensor(measurements)

        # Return subject ID string along with tensors for targeted logging
        return final_input, measurements_tensor, sub_id
