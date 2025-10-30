import numpy as np
import cv2
import torch
import open3d as o3d
import os
import glob
import pandas as pd
from ps6d_model import PS6DNetwork

# ============================================================
# PS6D Inference Class (Không đổi)
# ============================================================

class PS6DInference:
    def __init__(self, model_path, device='cuda', num_points=1024):
        self.device = device
        self.num_points = num_points
        self.model = PS6DNetwork(num_points=num_points, feature_dim=128)
        if not os.path.exists(model_path):
             print(f"LỖI: Không tìm thấy model checkpoint tại {model_path}")
             raise FileNotFoundError(model_path)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Loaded PS6D model from {model_path}")

    def normalize_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        max_extent = np.max(np.abs(centered)) + 1e-6
        scale = 0.1 / max_extent
        normalized = centered * scale
        return normalized, scale, centroid

    def denormalize_centroid(self, normalized_centroid, scale, offset):
        return normalized_centroid / scale + offset

    def predict_centroid(self, points_3d):
        if len(points_3d) < 50:
            return None, 0.0
        points_norm, scale, offset = self.normalize_point_cloud(points_3d)
        if len(points_norm) > self.num_points:
            indices = np.random.choice(len(points_norm), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points_norm), self.num_points, replace=True)
        points_sampled = points_norm[indices]
        points_tensor = torch.FloatTensor(points_sampled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_offset, pred_quat = self.model(points_tensor)
        pred_centroids = points_tensor[0] + pred_offset[0]
        centroid_norm = pred_centroids.mean(dim=0).cpu().numpy()
        centroid = self.denormalize_centroid(centroid_norm, scale, offset)
        variance = pred_centroids.var(dim=0).mean().cpu().item()
        confidence = np.exp(-variance * 10)
        return centroid, confidence

# ============================================================
# Helper Function (Lấy từ Cell 3)
# ============================================================

def get_point_cloud_from_mask(mask_img, depth_img, depth_intr, color_intr, R, t):
    """
    Trích xuất point cloud trực tiếp từ mask đen trắng (uint8).
    """
    if mask_img.shape != depth_img.shape:
        mask_img = cv2.resize(mask_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    v, u = np.where(mask_img > 128) # Lấy tọa độ (row, col) tức là (v, u)
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

# ============================================================
# Camera Intrinsics (Không đổi)
# ============================================================

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

# ============================================================
# Main Pipeline (Có logic chọn)
# ============================================================

def main_pipeline_with_ps6d(model_path, base_path, mask_dir, output_csv="submission_ps6d.csv"):
    """
    Main pipeline tích hợp PS6D (phiên bản dùng MASK NPY + Logic chọn)

    Args:
        model_path: Đường dẫn đến model ps6d_best.pth
        base_path: Đường dẫn đến thư mục train/test (chứa rgb/ depth/)
        mask_dir: Đường dẫn đến thư mục yolact_result/ (chứa masks/)
        output_csv: Tên file submission
    """
    # Khởi tạo PS6D
    try:
        ps6d = PS6DInference(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    except FileNotFoundError:
        return

    # Lấy file ảnh
    rgb_files = sorted(glob.glob(os.path.join(base_path, "rgb", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(base_path, "depth", "*.png")))

    all_final_outputs = []

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        IMAGE_FILENAME = os.path.basename(rgb_path) # '0000.png'
        BASE_NAME = os.path.splitext(IMAGE_FILENAME)[0] # '0000'

        print(f"\n{'='*50}")
        print(f"ĐANG XỬ LÝ ẢNH: {IMAGE_FILENAME}")
        print(f"{'='*50}")

        # Load ảnh
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        if depth is None or img is None:
            print(f"  -> Lỗi: Không load được ảnh")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None))
            continue

        # *** SỬA ĐƯỜNG DẪN: Load file .NPY từ thư mục masks/ ***
        mask_stack_path = os.path.join(mask_dir, "masks", BASE_NAME + ".npy") # Tìm trong yolact_result/masks/

        if not os.path.exists(mask_stack_path):
            print(f"  -> Lỗi: Không tìm thấy MASK STACK tại {mask_stack_path}")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None))
            continue

        try:
            mask_stack = np.load(mask_stack_path)
        except Exception as e:
            print(f"  -> Lỗi khi load file .npy {mask_stack_path}: {e}")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None))
            continue

        if mask_stack.ndim != 3 or mask_stack.shape[0] == 0:
             print(f"  -> Lỗi: File .npy có shape không hợp lệ: {mask_stack.shape}")
             all_final_outputs.append((IMAGE_FILENAME, None, None, None))
             continue

        num_candidates = mask_stack.shape[0]
        print(f"Tìm thấy {num_candidates} ứng cử viên (mask) từ file .npy.")

        # Lặp qua từng MASK để tìm ứng cử viên
        candidate_results = []
        for i in range(num_candidates):
            try:
                mask_instance = mask_stack[i]

                if mask_instance.dtype == bool:
                    mask_img = (mask_instance).astype(np.uint8) * 255
                elif mask_instance.dtype == np.uint8:
                    mask_img = mask_instance
                else:
                    mask_img = (mask_instance > 0.5).astype(np.uint8) * 255

                # Trích xuất Point Cloud
                points_3d = get_point_cloud_from_mask(
                    mask_img, depth,
                    depth_intrinsics, color_intrinsics,
                    R_depth_to_color, t_depth_to_color
                )

                if len(points_3d) < 100:
                    print(f"  -> Mask {i}: Quá ít điểm ({len(points_3d)})")
                    continue

                # Chạy PS6D
                centroid, confidence = ps6d.predict_centroid(points_3d)

                if centroid is not None:
                    candidate_results.append({
                        'centroid': centroid,
                        'confidence': confidence,
                        'mask_index': i
                    })
                    print(f"  Mask {i}: Tâm PS6D = ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), "
                          f"Confidence = {confidence:.3f}")

            except Exception as e:
                print(f"  -> Lỗi khi xử lý mask {i}: {e}")

        # Chọn MASK tốt nhất
        if len(candidate_results) == 0:
            print("[KẾT QUẢ]: Không có ứng cử viên hợp lệ từ mask")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None))
            continue

        candidate_results.sort(key=lambda x: (x['centroid'][2], -x['confidence']))
        z_min = candidate_results[0]['centroid'][2]
        tie_group = [c for c in candidate_results if abs(c['centroid'][2] - z_min) < 0.005] # 5mm

        if len(tie_group) > 1:
            tie_group.sort(key=lambda x: x['confidence'], reverse=True)
            selected = tie_group[0]
            print(f"  -> Nhiều ứng cử viên gần nhau, chọn mask {selected['mask_index']} theo confidence.")
        else:
            selected = tie_group[0]
            print(f"  -> Chọn mask {selected['mask_index']} (gần nhất).")

        c = selected['centroid']
        print(f"\n--- [KẾT QUẢ CUỐI CÙNG CHO ẢNH {IMAGE_FILENAME}] ---")
        print(f"  Tâm dự đoán (x, y, z): ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
        print(f"  Confidence: {selected['confidence']:.3f}")
        all_final_outputs.append((IMAGE_FILENAME, c[0], c[1], c[2]))

    print("\n=== HOÀN TẤT QUÁ TRÌNH ===")

    # Save results
    df = pd.DataFrame(all_final_outputs, columns=['image_filename', 'x', 'y', 'z'])
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"Đã lưu kết quả ra file: {output_csv}")

    return df

# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    MODEL_PATH = "/content/checkpoints/ps6d_best.pth"
    BASE_DATA_PATH = "/content/drive/MyDrive/ViettelAIRace/train"
    # Sửa đường dẫn này trỏ đến thư mục yolact_result/ (chứa masks/)
    MASK_DIR = "/content/drive/MyDrive/ViettelAIRace/yolact_result_train"
    OUTPUT_CSV = "submission_ps6d.csv"

    # Run inference with PS6D
    results = main_pipeline_with_ps6d(
        model_path=MODEL_PATH,
        base_path=BASE_DATA_PATH,
        mask_dir=MASK_DIR,
        output_csv=OUTPUT_CSV
    )

    print(f"Hoàn tất inference. Đã lưu vào {OUTPUT_CSV}")