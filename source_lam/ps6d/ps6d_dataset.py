import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d
import glob

# Import model và loss để hàm train_ps6d có thể sử dụng
from ps6d_model import PS6DLoss, PS6DNetwork

# ============================================================
# PreprocessedDataset Class 
# ============================================================
class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        """
        Khởi tạo Dataset. Chỉ cần trỏ đến thư mục dữ liệu đã tiền xử lý.
        """
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        print(f"Đã tìm thấy {len(self.data_files)} file dữ liệu đã tiền xử lý.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.data_files[idx])
            return data
        except Exception as e:
            # print(f"Lỗi khi tải file {self.data_files[idx]}: {e}. Bỏ qua...")
            return None

# ============================================================
# Training Function 
# ============================================================
def train_ps6d(model, train_loader, val_loader, num_epochs, lr, device, save_dir):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    start_epoch = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_loss_t': [], 'val_loss_t': [],
        'train_loss_r': [], 'val_loss_r': []
    }

    model_path = os.path.join(save_dir, 'ps6d_best.pth')

    # Logic tải checkpoint (chỉ khi fine-tune)
    if lr < 0.001:
        if os.path.exists(model_path):
            print(f"Đang tải lại model tốt nhất từ: {model_path} để fine-tune...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                best_val_loss = checkpoint.get('loss', float('inf'))
                start_epoch = 0
                print(f"Tải thành công. Bắt đầu fine-tune {num_epochs} epoch mới (Val Loss cũ: {best_val_loss:.4f})")
            except Exception as e:
                print(f"Lỗi khi tải checkpoint: {e}. Huấn luyện từ đầu.")
        else:
            print(f"Không tìm thấy checkpoint tại {model_path} để fine-tune. Sẽ huấn luyện từ đầu.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    has_rotation_gt = True

    if has_rotation_gt:
        weight_t = 4.0 # Tỷ lệ 80%
        weight_r = 1.0  # Tỷ lệ 20%
        print(f"Đang bật loss (T: {weight_t}, R: {weight_r}) - Ưu tiên MCE 80%")
        criterion = PS6DLoss(weight_translation=weight_t, weight_rotation=weight_r).to(device)
    else:
        print("Đang tắt loss rotation (weight=0.0)")
        criterion = PS6DLoss(weight_translation=1.0, weight_rotation=0.0).to(device)

    print("🚀 Bắt đầu training...")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        # --- Training ---
        model.train()
        train_loss_total = 0.0
        train_loss_t = 0.0
        train_loss_r = 0.0
        num_train_batches = 0

        train_desc = f"Epoch {epoch+1}/{num_epochs} [Train]"
        for batch in tqdm(train_loader, desc=train_desc):
            if batch is None:
                continue

            points, gt_centroid, gt_offset, gt_normal = batch

            points = points.to(device)
            gt_centroid = gt_centroid.to(device)
            gt_offset = gt_offset.to(device)
            gt_normal = gt_normal.to(device)

            pred_offset, pred_normal = model(points)

            loss, loss_dict = criterion(
                pred_offset, pred_normal,
                gt_offset, gt_normal,
                points, gt_centroid
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_loss_t += loss_dict['loss_translation']
            train_loss_r += loss_dict['loss_rotation']
            num_train_batches += 1

        avg_train_loss = train_loss_total / num_train_batches if num_train_batches > 0 else 0
        avg_train_loss_t = train_loss_t / num_train_batches if num_train_batches > 0 else 0
        avg_train_loss_r = train_loss_r / num_train_batches if num_train_batches > 0 else 0

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_loss_t = 0.0
        val_loss_r = 0.0
        num_val_batches = 0

        val_desc = f"Epoch {epoch+1}/{num_epochs} [Val]"
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=val_desc):
                if batch is None:
                    continue

                points, gt_centroid, gt_offset, gt_normal = batch

                points = points.to(device)
                gt_centroid = gt_centroid.to(device)
                gt_offset = gt_offset.to(device)
                gt_normal = gt_normal.to(device)

                pred_offset, pred_normal = model(points)

                loss, loss_dict = criterion(
                    pred_offset, pred_normal,
                    gt_offset, gt_normal,
                    points, gt_centroid
                )

                val_loss_total += loss.item()
                val_loss_t += loss_dict['loss_translation']
                val_loss_r += loss_dict['loss_rotation']
                num_val_batches += 1

        avg_val_loss = val_loss_total / num_val_batches if num_val_batches > 0 else 0
        avg_val_loss_t = val_loss_t / num_val_batches if num_val_batches > 0 else 0
        avg_val_loss_r = val_loss_r / num_val_batches if num_val_batches > 0 else 0

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
              f"Train Loss: {avg_train_loss:.4f} (T: {avg_train_loss_t:.4f}, R: {avg_train_loss_r:.4f}) - "
              f"Val Loss: {avg_val_loss:.4f} (T: {avg_val_loss_t:.4f}, R: {avg_val_loss_r:.4f})")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_loss_t'].append(avg_train_loss_t)
        history['val_loss_t'].append(avg_val_loss_t)
        history['train_loss_r'].append(avg_train_loss_r)
        history['val_loss_r'].append(avg_val_loss_r)

        if avg_val_loss < best_val_loss and num_val_batches > 0:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'ps6d_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"  -> 🎉 Đã lưu model tốt nhất tại {save_path} (Val Loss: {best_val_loss:.4f})")

    return model, history