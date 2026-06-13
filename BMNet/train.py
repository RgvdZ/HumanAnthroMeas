import torch
import numpy as np
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, writer):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for i, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        global_step = (epoch - 1) * len(train_loader) + i
        writer.add_scalar('Loss/train_batch', loss.item(), global_step)

    writer.add_scalar('Loss/train_epoch', losses.avg, epoch)
    return losses.avg

def validate_split(loader, model, device, epoch, writer, split_name):
    """通用评估函数，用于计算指定数据划分的分位数指标（MAE, TP50, TP75, TP90）"""
    model.eval()
    all_abs_errors = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            #print(f"output shape = {outputs.shape}")
            targets = targets.cpu()

            abs_err = torch.abs(outputs - targets)
            all_abs_errors.append(abs_err)

    all_abs_errors = torch.cat(all_abs_errors, dim=0).numpy()

    # Calculate overall tracking averages across all subjects and physical measurements
    mae_per_meas = np.mean(all_abs_errors, axis=0)
    tp50_per_meas = np.percentile(all_abs_errors, 50, axis=0)
    tp75_per_meas = np.percentile(all_abs_errors, 75, axis=0)
    tp90_per_meas = np.percentile(all_abs_errors, 90, axis=0)

    overall_mae = np.mean(mae_per_meas)
    overall_tp50 = np.mean(tp50_per_meas)
    overall_tp75 = np.mean(tp75_per_meas)
    overall_tp90 = np.mean(tp90_per_meas)

    # Stacking logs with matching suffixes so TensorBoard overlays them beautifully
    writer.add_scalar(f'Loss/Val_{split_name}_MAE', overall_mae, epoch)
    writer.add_scalar(f'Quantiles/Val_{split_name}_TP50', overall_tp50, epoch)
    writer.add_scalar(f'Quantiles/Val_{split_name}_TP75', overall_tp75, epoch)
    writer.add_scalar(f'Quantiles/Val_{split_name}_TP90', overall_tp90, epoch)

    return {
        'mae': overall_mae,
        'tp50': overall_tp50,
        'tp75': overall_tp75,
        'tp90': overall_tp90
    }

def print_single_subject(val_loader, model, device, epoch, measurement_labels, target_subject_idx=0):
    """在每个第10个Epoch，抽取并输出选定样本所有14个维度的回归误差"""
    model.eval()
    current_idx = 0

    with torch.no_grad():
        for inputs, targets, sub_ids in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            targets = targets.numpy()

            for b in range(inputs.size(0)):
                if current_idx == target_subject_idx:
                    print(f"\n====================== [EPOCH {epoch}] TARGET SUBJECT MONITORING ======================")
                    print(f" Subject ID: {sub_ids[b]}")
                    print(f"----------------------------------------------------------------------------------------")
                    for idx, name in enumerate(measurement_labels):
                        p_val = outputs[b][idx]
                        t_val = targets[b][idx]
                        err = abs(p_val - t_val)
                        print(f" -> {name.upper():<20} | Pred: {p_val:7.2f} | Target: {t_val:7.2f} | AbsError: {err:5.2f}")
                    print("========================================================================================\n")
                    return
                current_idx += 1

def fit(model, train_loader, val_loader, test_a_loader, test_b_loader, optimizer, criterion, device, start_epoch, total_epochs, measurement_labels, scheduler):
    writer = SummaryWriter(log_dir='./runs/bmnet_mnasnet_experiment')

    print(f"Starting training pipeline on {device} from Epoch {start_epoch}...")

    for epoch in range(start_epoch, total_epochs + 1):
        # 1. Train Step
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, writer)
        writer.add_scalar('Loss/Train_Epoch_MAE', train_loss, epoch)

        # 2. Evaluation Steps across all validation and test splits
        val_metrics = validate_split(val_loader, model, device, epoch, writer, split_name="Validation")
        test_a_metrics = validate_split(test_a_loader, model, device, epoch, writer, split_name="Test-A")
        test_b_metrics = validate_split(test_b_loader, model, device, epoch, writer, split_name="Test-B")

        # 3. Step learning rate optimization scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Console Progress Update
        print(f"Epoch [{epoch}/{total_epochs}] | LR: {current_lr:.6f} | Train MAE: {train_loss:.2f} || "
              f"Val MAE: {val_metrics['mae']:.2f} (TP90: {val_metrics['tp90']:.2f}) || "
              f"Test-A MAE: {test_a_metrics['mae']:.2f} || Test-B MAE: {test_b_metrics['mae']:.2f}")

        # Task requirement: Print 14 specific measurements every 10th epoch of exactly 1 subject
        if epoch % 10 == 0:
            print_single_subject(val_loader, model, device, epoch, measurement_labels, target_subject_idx=0)

        # Task requirement: Save checkpoint every 50th epoch [cite: 274]
        if epoch % 50 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
            }, epoch=epoch)

    writer.close()
    print("Training Complete.")
