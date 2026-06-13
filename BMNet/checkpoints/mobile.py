import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MNASNet1_0_Weights
import os

def load_model(checkpoint_path, device):
    """Create model and load weights."""
    weights = MNASNet1_0_Weights.DEFAULT
    model = models.mnasnet1_0(weights=weights)
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=True),
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 14)
    )
    
    # ? FIX: Add weights_only=False to allow loading numpy arrays from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict")
    
    model.eval()
    return model

def convert_to_torchscript(model, output_path, device):
    """Convert to TorchScript."""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 640, 960).to(device)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    traced_model.save(output_path)
    print(f"TorchScript model saved to: {output_path}")

if __name__ == "__main__":
    CHECKPOINT_FILE = "bmnet_mnasnet_weights_B.pth"
    OUTPUT_FILE = "bmnet_mobile.pt"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_FILE}")
    
    model = load_model(CHECKPOINT_FILE, device=DEVICE)
    convert_to_torchscript(model, output_path=OUTPUT_FILE, device=DEVICE)
    
    print("Done! Mobile model is ready.")