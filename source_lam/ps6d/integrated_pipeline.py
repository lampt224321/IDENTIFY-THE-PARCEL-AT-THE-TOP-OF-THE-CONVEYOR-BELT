import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import os
import glob
import pandas as pd
from ps6d_model import PS6DNetwork
from sklearn.cluster import DBSCAN

# ============================================================
# PS6D Inference Class 
# ============================================================
class PS6DInference:
    # ... (Giữ nguyên toàn bộ class PS6DInference, không thay đổi gì) ...
    def __init__(self, model_path, device='cuda', num_points=2048):
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
    def predict_pose(self, points_3d):
        if len(points_3d) < 50:
            return None, None, 0.0
        points_norm, scale, offset = self.normalize_point_cloud(points_3d)
        if len(points_norm) > self.num_points:
            indices = np.random.choice(len(points_norm), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points_norm), self.num_points, replace=True)
        points_sampled = points_norm[indices]
        points_tensor = torch.FloatTensor(points_sampled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_offset, pred_normal = self.model(points_tensor)
        pred_centroids_torch = points_tensor[0] + pred_offset[0]
        pred_normals_torch = pred_normal[0]
        pred_centroids_np = pred_centroids_torch.cpu().numpy()
        centroid_norm = None
        normal_vector_norm = None
        variance = 1e6
        try:
            clustering = DBSCAN(eps=0.01, min_samples=50).fit(pred_centroids_np)
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(counts) > 0:
                largest_cluster_label = unique_labels[counts.argmax()]
                inlier_mask = (labels == largest_cluster_label)
                inlier_votes_centroids = pred_centroids_torch[inlier_mask]
                inlier_votes_normals = pred_normals_torch[inlier_mask]
                centroid_norm_torch = inlier_votes_centroids.mean(dim=0)
                variance_torch = inlier_votes_centroids.var(dim=0).mean()
                normal_vector_norm_torch = inlier_votes_normals.mean(dim=0)
                normal_vector_norm_torch = F.normalize(normal_vector_norm_torch, p=2, dim=-1)
                centroid_norm = centroid_norm_torch.cpu().numpy()
                normal_vector_norm = normal_vector_norm_torch.cpu().numpy()
                variance = variance_torch.cpu().item()
            else:
                centroid_norm = pred_centroids_torch.mean(dim=0).cpu().numpy()
                normal_vector_norm_torch = F.normalize(pred_normals_torch.mean(dim=0), p=2, dim=-1)
                normal_vector_norm = normal_vector_norm_torch.cpu().numpy()
                variance = pred_centroids_torch.var(dim=0).mean().cpu().item()
        except Exception:
            centroid_norm = pred_centroids_torch.mean(dim=0).cpu().numpy()
            normal_vector_norm_torch = F.normalize(pred_normals_torch.mean(dim=0), p=2, dim=-1)
            normal_vector_norm = normal_vector_norm_torch.cpu().numpy()
            variance = pred_centroids_torch.var(dim=0).mean().cpu().item()
        centroid = self.denormalize_centroid(centroid_norm, scale, offset)
        confidence = np.exp(-variance * 10)
        return centroid, normal_vector_norm, confidence

# ============================================================
# HELPER FUNCTIONS 
# ============================================================
def project_3d_to_2d(p3d, intr):
    x, y, z = p3d[0], p3d[1], p3d[2]
    if z < 1e-5:
        return None
    u = (x * intr['fx'] / z) + intr['cx']
    v = (y * intr['fy'] / z) + intr['cy']
    return np.array([u, v])

# === HÀM ĐÃ SỬA LỖI LOGIC (GIỐNG NHƯ TRONG PREPROCESS.PY) ===
def get_point_cloud_from_mask(
    mask_img, 
    depth_img, 
    color_intr, # <<< CHỈ CẦN THÔNG SỐ MÀU
    depth_scale=1000.0, 
    z_min=0.01, 
    z_max=5.0
):
    if mask_img.shape != depth_img.shape:
        mask_img = cv2.resize(mask_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    v, u = np.where(mask_img > 128)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)

    Z = depth_img[v, u].astype(np.float32) / depth_scale
    valid = (Z > z_min) & (Z < z_max) & np.isfinite(Z)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    u, v, Z = u[valid], v[valid], Z[valid]

    # SỬA LỖI: Dùng thông số camera MÀU ('color_intr')
    X = (u - color_intr['cx']) * Z / color_intr['fx']
    Y = (v - color_intr['cy']) * Z / color_intr['fy']
    
    points_color_space = np.stack((X, Y, Z), axis=-1)
    return points_color_space
# === KẾT THÚC SỬA LỖI ===

# ============================================================
# Camera Intrinsics & ROI 
# ============================================================
ROI_BOX = (560, 150, 300, 330)
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
# Main Pipeline 
# ============================================================
def main_pipeline_with_ps6d(model_path, base_path, mask_dir, output_csv="Submission3D.csv"):

    np.random.seed(42)
    try:
        ps6d = PS6DInference(model_path, device='cuda' if torch.cuda.is_available() else 'cpu', num_points=2048)
    except FileNotFoundError:
        return

    rgb_files = sorted(glob.glob(os.path.join(base_path, "rgb", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(base_path, "depth", "*.png")))
    all_final_outputs = []

    u_min, v_min, w, h = ROI_BOX
    u_max = u_min + w
    v_max = v_min + h
    print(f"Đã kích hoạt ROI 2D: u=[{u_min}, {u_max}), v=[{v_min}, {v_max})")

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        IMAGE_FILENAME = os.path.basename(rgb_path)
        BASE_NAME = os.path.splitext(IMAGE_FILENAME)[0]

        print(f"\n{'='*50}")
        print(f"ĐANG XỬ LÝ ẢNH: {IMAGE_FILENAME}")
        print(f"{'='*50}")

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        if depth is None or img is None:
            print(f"  -> Lỗi: Không load được ảnh")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue

        H, W = depth.shape[:2]

        mask_search_pattern = os.path.join(mask_dir, "masks", BASE_NAME + "_*.npy")
        individual_mask_files = sorted(glob.glob(mask_search_pattern))

        if not individual_mask_files:
            print(f"  -> Lỗi: Không tìm thấy file MASK NPY nào với mẫu {mask_search_pattern}")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue

        print(f"Tìm thấy {len(individual_mask_files)} mask .npy.")
        candidate_results = []
        roi_mask_img = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(roi_mask_img, (u_min, v_min), (u_max, v_max), 255, -1)
        MIN_CONFIDENCE = 0.5

        for i, mask_file_path in enumerate(individual_mask_files):
            try:
                mask_instance = np.load(mask_file_path)
                if mask_instance.ndim != 2:
                    print(f"  -> Lỗi: File .npy {mask_file_path} không phải 2D. Bỏ qua.")
                    continue

                mask_img = (mask_instance > 0.5).astype(np.uint8) * 255

                if mask_img.shape[0] != H or mask_img.shape[1] != W:
                    mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

                # Lọc Overlap 
                total_pixels = np.sum(mask_img > 128)
                if total_pixels < 100: continue
                intersection = cv2.bitwise_and(mask_img, roi_mask_img)
                intersection_pixels = np.sum(intersection > 128)
                overlap_ratio = intersection_pixels / total_pixels
                MIN_OVERLAP_RATIO = 0.4
                if overlap_ratio < MIN_OVERLAP_RATIO:
                    print(f"  -> File {os.path.basename(mask_file_path)}: Bỏ qua (Overlap {overlap_ratio*100:.1f}% < {MIN_OVERLAP_RATIO*100}%)")
                    continue

                # === SỬA LỖI LỜI GỌI HÀM ===
                # Dòng CŨ:
                # points_3d = get_point_cloud_from_mask(
                #     mask_img, depth,
                #     depth_intrinsics, color_intrinsics,
                #     R_depth_to_color, t_depth_to_color
                # )
                
                # Dòng MỚI (đã sửa):
                points_3d = get_point_cloud_from_mask(
                    mask_img, depth,
                    color_intrinsics # Chỉ cần thông số MÀU
                )
                # === KẾT THÚC SỬA LỖI ===

                if len(points_3d) < 100:
                    # Thêm log để biết TẠI SAO nó bị lọc
                    print(f"  -> File {os.path.basename(mask_file_path)}: Bỏ qua (Không đủ điểm 3D: {len(points_3d)} < 100)")
                    continue

                centroid, normal_vector, confidence = ps6d.predict_pose(points_3d)

                if centroid is None:
                    continue

                if confidence < MIN_CONFIDENCE:
                    print(f"  -> File {os.path.basename(mask_file_path)}: Bỏ qua (Confidence {confidence:.2f} < {MIN_CONFIDENCE})")
                    continue

                # Lọc tâm 2D 
                centroid_2d = project_3d_to_2d(centroid, color_intrinsics)
                if centroid_2d is None:
                    print(f"  -> File {os.path.basename(mask_file_path)}: Bỏ qua (Tâm 3D có Z=0)")
                    continue

                u_pred, v_pred = centroid_2d
                if not (u_min <= u_pred < u_max and v_min <= v_pred < v_max):
                    print(f"  -> File {os.path.basename(mask_file_path)}: Bỏ qua (Tâm 2D ({u_pred:.0f}, {v_pred:.0f}) nằm ngoài ROI, dù overlap đủ)")
                    continue

                candidate_results.append({
                    'centroid': centroid,
                    'normal_vector': normal_vector, 
                    'confidence': confidence,
                    'mask_index': i,
                    'file_name': os.path.basename(mask_file_path)
                })
                print(f"  File {os.path.basename(mask_file_path)}: ĐƯỢC CHỌN (Tâm PS6D = ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), Conf: {confidence:.2f})")

            except Exception as e:
                print(f"  -> Lỗi khi xử lý file mask {mask_file_path}: {e}")

        # ... (Toàn bộ phần logic chọn MASK tốt nhất và lưu file CSV được giữ nguyên) ...
        if len(candidate_results) == 0:
            print("[KẾT QUẢ]: Không có ứng cử viên hợp lệ (Tất cả đã bị lọc bởi ROI / Overlap / Confidence)")
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue
        candidate_results.sort(key=lambda x: x['centroid'][2])
        z_min = candidate_results[0]['centroid'][2]
        tie_group = [c for c in candidate_results if abs(c['centroid'][2] - z_min) < 0.005]
        if len(tie_group) > 1:
            tie_group.sort(key=lambda x: x['centroid'][0])
            selected = tie_group[0]
            print(f"  -> Nhiều ứng cử viên gần nhau, chọn file {selected['file_name']} theo tọa độ X (bé nhất).")
        else:
            selected = tie_group[0]
            print(f"  -> Chọn file {selected['file_name']} (gần nhất - Z min).")
        c = selected['centroid']
        n = selected['normal_vector']
        print(f"\n--- [KẾT QUẢ CUỐI CÙNG CHO ẢNH {IMAGE_FILENAME}] ---")
        print(f"  Tâm dự đoán (x, y, z): ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
        print(f"  Pháp tuyến (nx, ny, nz): ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
        print(f"  Confidence: {selected['confidence']:.3f}")
        all_final_outputs.append((IMAGE_FILENAME, c[0], c[1], c[2], n[0], n[1], n[2]))

    print("\n=== HOÀN TẤT QUÁ TRÌNH ===")
    df = pd.DataFrame(all_final_outputs, columns=['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"Đã lưu kết quả ra file: {output_csv}")
    return df

# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "/home/hp/VTAIRACE/source_lam/ps6d/ps6dmodel/testmodel/ps6d_best.pth"
    BASE_DATA_PATH = "/home/hp/VTAIRACE/source_lam/dataset/Train"
    MASK_DIR = "/home/hp/VTAIRACE/Code_Angie/masks_npy"
    OUTPUT_CSV = "Submission3D.csv"

    results = main_pipeline_with_ps6d(
        model_path=MODEL_PATH,
        base_path=BASE_DATA_PATH,
        mask_dir=MASK_DIR,
        output_csv=OUTPUT_CSV
    )
    print(f"Hoàn tất inference. Đã lưu vào {OUTPUT_CSV}")