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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_point_cloud_from_mask(
    mask_img, 
    depth_img, 
    color_intr, # <<< SỬA LỖI LOGIC: Dùng thông số camera MÀU
    depth_scale=1000.0, 
    z_min=0.01, 
    z_max=5.0
):
    """
    Tạo point cloud từ mask (ảnh màu) và depth map (đã đăng ký).
    """
    if mask_img.shape != depth_img.shape:
        mask_img = cv2.resize(mask_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    v, u = np.where(mask_img > 128)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Lấy giá trị Z (độ sâu) tại các pixel (u,v)
    Z = depth_img[v, u].astype(np.float32) / depth_scale
    
    # Lọc các giá trị Z không hợp lệ
    valid = (Z > z_min) & (Z < z_max) & np.isfinite(Z)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32) # Trả về 0 nếu không có điểm hợp lệ

    u, v, Z = u[valid], v[valid], Z[valid]

    # Tọa độ (u, v) đến từ MASK (ảnh MÀU), 
    # vì vậy chúng ta phải dùng thông số của camera MÀU ('color_intr')
    X = (u - color_intr['cx']) * Z / color_intr['fx']
    Y = (v - color_intr['cy']) * Z / color_intr['fy']

    # Đám mây điểm 3D này đã ở trong hệ tọa độ của camera MÀU.
    points_color_space = np.stack((X, Y, Z), axis=-1)

    return points_color_space


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    max_extent = np.max(np.abs(centered)) + 1e-6
    scale = 0.1 / max_extent
    normalized = centered * scale
    return normalized, scale, centroid

def project_3d_to_2d(p3d, intr):
    x, y, z = p3d[0], p3d[1], p3d[2]
    if z < 1e-5:
        return None
    u = (x * intr['fx'] / z) + intr['cx']
    v = (y * intr['fy'] / z) + intr['cy']
    return np.array([u, v])

# ============================================================
#  CLASS ParcelDataset
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
        self.has_rotation_gt = 'Rx' in self.gt_df.columns

    def __len__(self):
        return len(self.gt_df)

    def __getitem__(self, idx):
        try:
            # 1. Lấy Ground Truth
            gt_row = self.gt_df.iloc[idx]
            image_name_gt = gt_row['image_filename'] # Dùng để debug
            gt_centroid_world = np.array([gt_row['x'], gt_row['y'], gt_row['z']], dtype=np.float32)

            # 2. Chiếu GT 3D sang 2D
            gt_centroid_2d = project_3d_to_2d(gt_centroid_world, self.color_intr)
            if gt_centroid_2d is None:
                # print(f"LOG (idx {idx}, {image_name_gt}): Bỏ qua, GT 3D chiếu ra None")
                return None

            # === SỬA LỖI TÊN FILE ===
            # Giả định idx=0 -> 0000.png, idx=81 -> 0081.png
            base_name_for_files = f"{idx:04d}" 
            actual_image_filename = f"{base_name_for_files}.png"
            # === KẾT THÚC SỬA LỖI ===

            # Lấy GT Rotation
            if self.has_rotation_gt:
                gt_normal_vector = np.array([gt_row['Rx'], gt_row['Ry'], gt_row['Rz']], dtype=np.float32)
                gt_normal_norm = gt_normal_vector / (np.linalg.norm(gt_normal_vector) + 1e-8)
            else:
                gt_normal_norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # 3. Load ảnh
            depth_path = os.path.join(self.depth_dir, actual_image_filename)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                print(f"LỖI (idx {idx}, {actual_image_filename}): Không tìm thấy file depth tại {depth_path}")
                return None

            H, W = depth_img.shape[:2]

            # 4. Tìm các mask .NPY (Ví dụ: .../0000_*.npy)
            mask_search_pattern = os.path.join(self.npy_mask_dir, base_name_for_files + "_*.npy")
            candidate_mask_files = sorted(glob.glob(mask_search_pattern))

            if not candidate_mask_files:
                print(f"LỖI (idx {idx}, {actual_image_filename}): Không tìm thấy file mask nào với mẫu {mask_search_pattern}")
                return None

            # 5. Logic ghép cặp
            best_mask_img = None
            u_gt = int(round(gt_centroid_2d[0]))
            v_gt = int(round(gt_centroid_2d[1]))
            if not (0 <= v_gt < H and 0 <= u_gt < W):
                print(f"LỖI (idx {idx}, {actual_image_filename}): GT 2D ({u_gt}, {v_gt}) nằm ngoài ảnh ({W}, {H})")
                return None

            primary_candidates = []
            fallback_candidates = []
            radius = 1 
            v_start = max(0, v_gt - radius); v_end = min(H, v_gt + radius + 1)
            u_start = max(0, u_gt - radius); u_end = min(W, u_gt + radius + 1)
            gt_centroid_2d_flat = np.array([u_gt, v_gt])

            for mask_path in candidate_mask_files:
                try:
                    mask_instance = np.load(mask_path)
                    if mask_instance.ndim != 2: continue

                    mask_img = (mask_instance > 0.5).astype(np.uint8) * 255

                    if mask_img.shape[0] != H or mask_img.shape[1] != W:
                         mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

                    M = cv2.moments(mask_img)
                    if M["m00"] == 0: continue

                    mask_centroid_2d = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                    dist_2d = np.linalg.norm(mask_centroid_2d - gt_centroid_2d_flat)

                    window = mask_img[v_start:v_end, u_start:u_end]
                    if np.any(window > 128):
                        primary_candidates.append({'mask': mask_img, 'dist': dist_2d})
                    else:
                        fallback_candidates.append({'mask': mask_img, 'dist': dist_2d})
                except Exception:
                    continue

            # 6. Quyết định logic chọn
            if primary_candidates:
                primary_candidates.sort(key=lambda x: x['dist'])
                best_mask_img = primary_candidates[0]['mask']
            elif fallback_candidates:
                fallback_candidates.sort(key=lambda x: x['dist'])
                best_mask_img = fallback_candidates[0]['mask']
            else:
                print(f"LỖI (idx {idx}, {actual_image_filename}): Không có mask nào khớp (cả chính và phụ) với GT 2D ({u_gt}, {v_gt})")
                return None

            # 7. Trích xuất Point Cloud (ĐÃ SỬA LỖI LOGIC)
            points_3d = get_point_cloud_from_mask(
                best_mask_img, depth_img,
                self.color_intr # Chỉ cần truyền thông số MÀU
            )
            if len(points_3d) < 50:
                print(f"LỖI (idx {idx}, {actual_image_filename}): Không đủ điểm 3D ({len(points_3d)} < 50). Khả năng cao file depth hỏng (toàn 0).")
                return None

            # 8. Chuẩn hóa
            gt_centroid_norm = gt_centroid_world
            if self.normalize:
                points_3d, scale, offset = normalize_point_cloud(points_3d)
                gt_centroid_norm = (gt_centroid_world - offset) * scale

            # 9. Sample/Pad điểm
            if len(points_3d) > self.num_points:
                indices = np.random.choice(len(points_3d), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(points_3d), self.num_points, replace=True)
            points_sampled = points_3d[indices]

            # 10. Tính toán GT offset
            gt_centroid_norm_expanded = np.expand_dims(gt_centroid_norm, axis=0)
            gt_offset = gt_centroid_norm_expanded - points_sampled

            # 11. Xử lý GT Rotation
            gt_normal_per_point = np.tile(gt_normal_norm, (self.num_points, 1))

            return (
                torch.from_numpy(points_sampled.astype(np.float32)),
                torch.from_numpy(gt_centroid_norm.astype(np.float32)),
                torch.from_numpy(gt_offset.astype(np.float32)),
                torch.from_numpy(gt_normal_per_point.astype(np.float32))
            )

        except Exception as e:
            # print(f"LỖI CHƯA BIẾT (idx {idx}): {e}")
            return None # Bật lại try...except

# Hàm collate_fn
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

# ============================================================
# MAIN SCRIPT (ĐỂ CHẠY PREPROCESSING)
# ============================================================

def main_preprocess():
    print("Bắt đầu quá trình Tiền xử lý (Preprocessing)...")

    base_path = r"/home/hp/VTAIRACE/source_lam/dataset/Train"
    mask_dir = r"/home/hp/VTAIRACE/Code_Angie/masks_npy" 
    gt_csv_path = r"/home/hp/VTAIRACE/source_lam/dataset/Train/Public_train.csv"
    NUM_POINTS = 2048
    OUTPUT_DIR = "/home/hp/VTAIRACE/source_lam/ps6d/preprocessed_data"

    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Thư mục {OUTPUT_DIR} đã tồn tại. Xóa các file cũ...")
        for f in glob.glob(os.path.join(OUTPUT_DIR, "*.pt")):
            os.remove(f)
    else:
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")

    color_intrinsics = {
        'width': 1280, 'height': 720,
        'fx': 643.90087890625, 'fy': 643.1365356445312,
        'cx': 650.2113037109375, 'cy': 355.79559326171875,
    }
    depth_intrinsics = {
        'width': 1280, 'height': 720,
        'fx': 650.0616455078125, 'fy': 650.0616455078125,
        'cx': 649.5928955078125, 'cy': 360.9415588378906,
    }
    R_depth_to_color = np.array([
        [0.9999898076057434, -0.00020347206736914814, -0.004507721401751041],
        [0.00018898719281423837, 0.9999948143959045, -0.0032135415822267532],
        [0.004508351907134056, 0.003212657058611512, 0.9999846816062927]
    ])
    t_depth_to_color = np.array([[-0.05905], [8.67399e-5], [0.00041]])

    # 1. Khởi tạo Dataset 
    dataset = ParcelDataset(
        base_path=base_path,
        mask_dir=mask_dir,
        gt_csv_path=gt_csv_path,
        num_points=NUM_POINTS,
        color_intrinsics=color_intrinsics,
        depth_intrinsics=depth_intrinsics,
        R_d2c=R_depth_to_color,
        t_d2c=t_depth_to_color,
        normalize=True
    )

    # 2. Bật lại num_workers để chạy nhanh
    try:
        cpu_count = os.cpu_count()
        num_workers = max(1, cpu_count // 2)
    except:
        num_workers = 8

    print(f"Sử dụng {num_workers} workers để tiền xử lý {len(dataset)} ảnh...")

    loader = DataLoader(dataset, batch_size=8, num_workers=num_workers, collate_fn=collate_fn)

    save_idx = 0
    # Thêm 'total' vào tqdm để nó biết tổng số batch
    for batch_data in tqdm(loader, desc="Đang tiền xử lý", total=len(loader)):
        if batch_data is None:
            continue

        points_batch, centroid_batch, offset_batch, normal_batch = batch_data

        for i in range(points_batch.size(0)):
            sample = (
                points_batch[i],
                centroid_batch[i],
                offset_batch[i],
                normal_batch[i]
            )
            torch.save(sample, f"{OUTPUT_DIR}/sample_{save_idx}.pt")
            save_idx += 1

    print(f"\n✅ Tiền xử lý hoàn tất! Đã lưu {save_idx} file dữ liệu 'sạch' vào {OUTPUT_DIR}")

if __name__ == "__main__":
    main_preprocess()