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

from ps6d_model import PS6DLoss, PS6DNetwork

# ============================================================
# Helper Functions
# ============================================================

def get_point_cloud_from_mask(mask_img, depth_img, depth_intr, color_intr, R, t):
    """
    Trích xuất point cloud trực tiếp từ mask đen trắng (uint8).
    """
    if mask_img.shape != depth_img.shape:
        mask_img = cv2.resize(mask_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    v, u = np.where(mask_img > 128)
    if len(u) == 0:
        return np.array([])
    Z_values = depth_img[v, u]
    Z = Z_values.astype(np.float32) / 1000.0
    valid_depth = (Z > 0.01) & (Z < 5.0) & np.isfinite(Z)
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]
    Z_valid = Z[valid_depth]
    if len(u_valid) == 0:
        return np.array([])
    X = (u_valid - depth_intr['cx']) * Z_valid / depth_intr['fx']
    Y = (v_valid - depth_intr['cy']) * Z_valid / depth_intr['fy']
    points_depth_valid = np.stack((X, Y, Z_valid), axis=-1)
    points_color_valid = (R @ points_depth_valid.T).T + t.reshape(1, 3)
    return points_color_valid

def normalize_point_cloud(points):
    """
    Chuẩn hóa đám mây điểm (Không đổi)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    max_extent = np.max(np.abs(centered)) + 1e-6
    scale = 0.1 / max_extent
    normalized = centered * scale
    return normalized, scale, centroid

def project_3d_to_2d(p3d, intr):
    """
    Chiếu một điểm 3D (trong hệ tọa độ camera màu) sang tọa độ pixel 2D.
    """
    x, y, z = p3d[0], p3d[1], p3d[2]
    if z < 1e-5: # Tránh chia cho 0 hoặc giá trị quá nhỏ
        return None
    u = (x * intr['fx'] / z) + intr['cx']
    v = (y * intr['fy'] / z) + intr['cy']
    return np.array([u, v])

# ============================================================
# ParcelDataset Class (CẬP NHẬT LOGIC __getitem__)
# ============================================================

class ParcelDataset(Dataset):
    def __init__(self, base_path, mask_dir, gt_csv_path, num_points,
                 color_intrinsics, depth_intrinsics, R_d2c, t_d2c, normalize=True):

        self.base_path = base_path
        self.mask_dir = mask_dir 
        self.gt_df = pd.read_csv(gt_csv_path)
        self.num_points = num_points
        self.color_intr = color_intrinsics
        self.depth_intr = depth_intrinsics
        self.R_d2c = R_d2c
        self.t_d2c = t_d2c
        self.normalize = normalize
        self.rgb_dir = os.path.join(base_path, "rgb")
        self.depth_dir = os.path.join(base_path, "depth")
        self.npy_mask_dir = os.path.join(self.mask_dir, "masks")

        if 'Rx' not in self.gt_df.columns:
            print("Cảnh báo: Không tìm thấy cột 'Rx' trong GT. Sẽ sử dụng rotation giả lập.")
            self.has_rotation_gt = False
        else:
            self.has_rotation_gt = True
            print("Đã phát hiện GT rotation (Rx, Ry, Rz). Sẽ huấn luyện với loss rotation.")

    def __len__(self):
        return len(self.gt_df)

    def __getitem__(self, idx):
        gt_row = None
        try:
            # 1. Lấy Ground Truth
            gt_row = self.gt_df.iloc[idx]
            image_name_gt = gt_row['image_filename']
            gt_centroid_world = np.array([gt_row['x'], gt_row['y'], gt_row['z']], dtype=np.float32)

            # *** BƯỚC 1: Chiếu GT 3D sang 2D ***
            gt_centroid_2d = project_3d_to_2d(gt_centroid_world, self.color_intr)
            if gt_centroid_2d is None:
                print(f"Warning: GT centroid 3D có Z=0 cho {image_name_gt}. Bỏ qua...")
                return None

            if image_name_gt.startswith('image_'):
                actual_image_filename = image_name_gt[len('image_'):]
            else:
                actual_image_filename = image_name_gt
            base_name_for_files = os.path.splitext(actual_image_filename)[0]

            # Lấy GT Rotation (Không đổi)
            if self.has_rotation_gt:
                euler_angles = [gt_row['Rx'], gt_row['Ry'], gt_row['Rz']]
                r = R_scipy.from_euler('xyz', euler_angles, degrees=False)
                gt_quat_scipy = r.as_quat()
                gt_quat = np.array([
                    gt_quat_scipy[3], gt_quat_scipy[0], gt_quat_scipy[1], gt_quat_scipy[2]
                ], dtype=np.float32)
            else:
                gt_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # 2. Load ảnh (Không đổi)
            depth_path = os.path.join(self.depth_dir, actual_image_filename)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                print(f"Warning: Không thể load ảnh {depth_path}. Bỏ qua...")
                return None
            
            H, W = depth_img.shape[:2] # Lấy kích thước ảnh (720, 1280)

            
            # 3. Tìm tất cả các mask .NPY ứng cử viên
            mask_search_pattern = os.path.join(self.npy_mask_dir, base_name_for_files + "_*.npy")
            candidate_mask_files = sorted(glob.glob(mask_search_pattern))

            if not candidate_mask_files:
                print(f"Warning: Không tìm thấy file MASK NPY nào cho {base_name_for_files}. Bỏ qua...")
                return None

            # 4. Tìm mask .NPY "đúng" bằng cách kiểm tra "Point-in-Mask"
            best_mask_img = None
            
            # Lấy tọa độ GT 2D (dưới dạng số nguyên)
            u_gt = int(round(gt_centroid_2d[0]))
            v_gt = int(round(gt_centroid_2d[1]))
            
            # Kiểm tra xem tọa độ GT có nằm trong ảnh không
            if not (0 <= v_gt < H and 0 <= u_gt < W):
                print(f"Warning: Tọa độ GT 2D ({u_gt}, {v_gt}) nằm ngoài ảnh {base_name_for_files}. Bỏ qua...")
                return None

            for mask_path in candidate_mask_files:
                try:
                    mask_instance = np.load(mask_path)
                    if mask_instance.ndim != 2: continue

                    if mask_instance.dtype == bool:
                        mask_img = (mask_instance).astype(np.uint8) * 255
                    else:
                        mask_img = (mask_instance > 0.5).astype(np.uint8) * 255
                    
                    # Resize mask về kích thước chuẩn của ảnh depth
                    if mask_img.shape[0] != H or mask_img.shape[1] != W:
                         mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

                    # *** LOGIC MỚI: Kiểm tra pixel tại (v_gt, u_gt) ***
                    if mask_img[v_gt, u_gt] > 128:
                        # Tìm thấy! Mask này chứa điểm GT.
                        best_mask_img = mask_img 
                        break # Dừng tìm kiếm ngay lập tức

                except Exception as e:
                    print(f"Warning: Lỗi khi xử lý mask ứng cử viên {mask_path}: {e}")
                    continue

            # 5. Kiểm tra lại sau khi lặp
            if best_mask_img is None:
                print(f"Warning: Không tìm thấy mask NPY nào chứa GT 2D ({u_gt}, {v_gt}) cho {base_name_for_files}. Bỏ qua...")
                return None
            
            
            # 6. Trích xuất Point Cloud từ mask 2D TỐT NHẤT đã chọn
            points_3d = get_point_cloud_from_mask(
                best_mask_img, depth_img,
                self.depth_intr, self.color_intr, self.R_d2c, self.t_d2c
            )
            
            if len(points_3d) < 50:
                print(f"Warning: Quá ít điểm ({len(points_3d)}) cho {image_name_gt} từ MASK NPY đã chọn. Bỏ qua...")
                return None

            # 7. Chuẩn hóa (Normalize)
            gt_centroid_norm = gt_centroid_world
            if self.normalize:
                points_3d, scale, offset = normalize_point_cloud(points_3d)
                gt_centroid_norm = (gt_centroid_world - offset) * scale

            # 8. Sample/Pad điểm
            if len(points_3d) > self.num_points:
                indices = np.random.choice(len(points_3d), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(points_3d), self.num_points, replace=True)
            points_sampled = points_3d[indices]

            # 9. Tính toán GT offset
            gt_centroid_norm_expanded = np.expand_dims(gt_centroid_norm, axis=0)
            gt_offset = gt_centroid_norm_expanded - points_sampled

            # 10. Xử lý GT Quaternion
            gt_quat_norm = gt_quat / (np.linalg.norm(gt_quat) + 1e-8)
            gt_quat_per_point = np.tile(gt_quat_norm, (self.num_points, 1))

            return (
                torch.from_numpy(points_sampled.astype(np.float32)),
                torch.from_numpy(gt_centroid_norm.astype(np.float32)),
                torch.from_numpy(gt_offset.astype(np.float32)),
                torch.from_numpy(gt_quat_per_point.astype(np.float32))
            )

        except Exception as e:
            item_name = gt_row['image_filename'] if gt_row is not None else f"idx {idx}"
            print(f"Lỗi nghiêm trọng khi xử lý {item_name}: {e}. Bỏ qua...")
            return None


def train_ps6d(model, train_loader, val_loader, num_epochs, lr, device, save_dir):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset_to_check = train_loader.dataset
    while isinstance(dataset_to_check, torch.utils.data.Subset):
        dataset_to_check = dataset_to_check.dataset
    has_rotation_gt = dataset_to_check.has_rotation_gt
    if has_rotation_gt:
        print("Đang bật loss rotation (weight=1.0)")
        criterion = PS6DLoss(weight_translation=1.0, weight_rotation=1.0).to(device)
    else:
        print("Đang tắt loss rotation (weight=0.0)")
        criterion = PS6DLoss(weight_translation=1.0, weight_rotation=0.0).to(device)
    best_val_loss = float('inf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("🚀 Bắt đầu training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        # --- Training ---
        model.train()
        train_loss_total = 0.0
        train_loss_t = 0.0
        train_loss_r = 0.0
        num_train_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            if batch is None:
                continue
            points, gt_centroid, gt_offset, gt_quat = batch
            points = points.to(device)
            gt_centroid = gt_centroid.to(device)
            gt_offset = gt_offset.to(device)
            gt_quat = gt_quat.to(device)
            pred_offset, pred_quat = model(points)
            loss, loss_dict = criterion(
                pred_offset, pred_quat,
                gt_offset, gt_quat,
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
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                if batch is None:
                    continue
                points, gt_centroid, gt_offset, gt_quat = batch
                points = points.to(device)
                gt_centroid = gt_centroid.to(device)
                gt_offset = gt_offset.to(device)
                gt_quat = gt_quat.to(device)
                pred_offset, pred_quat = model(points)
                loss, loss_dict = criterion(
                    pred_offset, pred_quat,
                    gt_offset, gt_quat,
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

    return model