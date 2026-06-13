import os
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch, save_dir="checkpoints", filename="bmnet_mnasnet_weights.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- MINIMUM MODIFICATION: Alternate between slot A and slot B ---
    slot = "A" if (epoch // 50) % 2 != 0 else "B"
    filename = filename.replace(".pth", f"_{slot}.pth")
    # -----------------------------------------------------------------

    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    print(f"--> Saved system checkpoint at Epoch {epoch} safely to {save_path}")
