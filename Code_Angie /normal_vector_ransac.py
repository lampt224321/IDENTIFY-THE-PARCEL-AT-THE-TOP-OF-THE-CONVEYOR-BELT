import pandas as pd
import open3d as o3d
import numpy as np
import os
import re  # Để trích xuất tên file
from tqdm import tqdm  # Để xem thanh tiến trình

# --- HẰNG SỐ CẤU HÌNH ---
CSV_PATH = "/Users/angelinacu/Desktop/Study/Viettel/ThiSinh/train/Public train.csv"
PLY_DIR = "/Users/angelinacu/Desktop/Study/Viettel/ThiSinh/train/ply/"
# Đổi tên file output mới
OUTPUT_CSV_PATH = "image_results_with_curvature.csv" 

# Ma trận chuyển đổi 4x4 cho open3d
TRANSFORM_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# --- THAM SỐ THUẬT TOÁN (CỐ ĐỊNH) ---
K_NEIGHBORS = 4000       # Cố định K_NEIGHBORS
RANSAC_DISTANCE_THRESHOLD = 0.008 # Cố định ngưỡng
RANSAC_N = 3             # Theo yêu cầu của bạn
RANSAC_ITERATIONS = 1000 # Cố định số lần lặp



def get_ply_path_from_image(image_filename, base_ply_dir):
    """
    Chuyển đổi tên file từ 'image_XXXX.png' thành đường dẫn '.../XXXX.ply'.
    """
    match = re.search(r'image_(\d+)\.png', image_filename)
    if not match:
        return None
    ply_name = f"{match.group(1)}.ply"
    return os.path.join(base_ply_dir, ply_name)

def calculate_normal_and_curvature(pcd, center_point, k, dist_thresh, n_pts, iters):
    """
    NÂNG CẤP: Tính normal vector VÀ surface variation (độ cong)
    1. Tìm k lân cận (KNN).
    2. Chạy RANSAC để tìm 'inliers'.
    3. Chạy PCA trên 'inliers'.
    4. Trả về pháp tuyến VÀ độ cong.
    """
    try:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k_found, idx, _] = pcd_tree.search_knn_vector_3d(center_point, k)
        
        # Cần ít nhất n_pts (4) điểm để chạy RANSAC
        if k_found < n_pts:
            return None, None

        neighbor_points = np.asarray(pcd.points)[idx, :]
        pcd_neighbors = o3d.geometry.PointCloud()
        pcd_neighbors.points = o3d.utility.Vector3dVector(neighbor_points)

        plane_model, inlier_indices = pcd_neighbors.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=n_pts,
            num_iterations=iters
        )
        
        inlier_points = neighbor_points[inlier_indices, :]
        
        # Cần ít nhất n_pts (4) điểm để tính PCA ổn định
        if inlier_points.shape[0] < n_pts:
            return None, None

        # --- TÍNH TOÁN PCA VÀ ĐỘ CONG ---
        covariance_matrix = np.cov(inlier_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # eigenvalues được trả về theo thứ tự tăng dần
        lambda_0 = eigenvalues[0] # Giá trị riêng nhỏ nhất
        lambda_1 = eigenvalues[1]
        lambda_2 = eigenvalues[2]
        
        sum_eigenvalues = lambda_0 + lambda_1 + lambda_2
        
        if sum_eigenvalues == 0:
            return None, None # Tránh chia cho 0

        # 1. Tính Surface Variation (Độ cong)
        surface_variation = lambda_0 / sum_eigenvalues
        
        # 2. Tính Normal Vector (vector riêng ứng với giá trị riêng nhỏ nhất)
        normal_vector = eigenvectors[:, 0] 
        
        return normal_vector, surface_variation

    except Exception as e:
        # print(f"Lỗi: {e}")
        return None, None

def compare_normals(calculated_normal, gt_normal):
    """
    Chuẩn hóa, so sánh hai vector và trả về góc lệch (độ).
    """
    norm_calc = np.linalg.norm(calculated_normal)
    if norm_calc == 0:
        return None
    calculated_normal /= norm_calc
    
    dot_product = np.dot(calculated_normal, gt_normal)
    cos_theta = np.abs(dot_product)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def main():
    # 1. Đọc file CSV chính
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file CSV tại: {CSV_PATH}")
        return

    # --- [THAY ĐỔI] ---
    # Danh sách lưu kết quả cuối cùng cho file CSV
    results_list = []
    # Danh sách chỉ lưu lỗi góc để tính trung bình
    all_angle_errors_list = []
    # --- [KẾT THÚC THAY ĐỔI] ---

    print(f"Bắt đầu xử lý (RANSAC+PCA) với K={K_NEIGHBORS}, N={RANSAC_N}, Thresh={RANSAC_DISTANCE_THRESHOLD}")
    
    # 2. Lặp qua từng hàng trong file CSV (từng mẫu)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Đang xử lý file"):
        image_name = row['image_filename']
        center_point = np.array([row['x'], row['y'], row['z']])
        gt_normal = np.array([row['Rx'], row['Ry'], row['Rz']])

        # 3. Tải và tiền xử lý
        ply_path = get_ply_path_from_image(image_name, PLY_DIR)
        if not ply_path or not os.path.exists(ply_path):
            continue

        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            if pcd.is_empty():
                continue
            pcd.transform(TRANSFORM_MATRIX)
        except Exception as e:
            continue

        # --- [THAY ĐỔI] ---
        # 4. Xóa bỏ vòng lặp so sánh ngưỡng
        # Gọi hàm tính toán 1 lần với các tham số cố định
        
        calculated_normal, surface_variation = calculate_normal_and_curvature(
            pcd, 
            center_point, 
            K_NEIGHBORS,
            RANSAC_DISTANCE_THRESHOLD,
            RANSAC_N, 
            RANSAC_ITERATIONS
        )
        
        # Nếu tính toán thất bại, bỏ qua
        if calculated_normal is None or surface_variation is None:
            continue

        angle_deg = compare_normals(calculated_normal, gt_normal)
        
        if angle_deg is not None:
            # Lưu kết quả vào danh sách
            results_list.append({
                'image_filename': image_name,
                'angle_error_deg': angle_deg,
                'surface_variation': surface_variation
            })
            # Lưu lỗi góc để tính trung bình
            all_angle_errors_list.append(angle_deg)
        # --- [KẾT THÚC THAY ĐỔI] ---

    # --- KẾT THÚC VÒNG LẶP CHÍNH ---

    # 5. Tính toán, tạo bảng so sánh và xuất file
    if not results_list:
        print("\nKhông xử lý thành công bất kỳ file nào.")
        return

    # A. Lưu file CSV chi tiết (Yêu cầu 1 của bạn)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
    print(f"\nĐã lưu kết quả chi tiết (lỗi góc, độ cong) vào: {OUTPUT_CSV_PATH}")

    # B. Tính và in kết quả trung bình ra terminal (Yêu cầu 2 của bạn)
    mean_angle_error = np.mean(all_angle_errors_list)
    
    print("\n" + "="*55)
    print(f"--- TỔNG KẾT (với K={K_NEIGHBORS}, N={RANSAC_N}, Thresh={RANSAC_DISTANCE_THRESHOLD}) ---")
    print(f"Tổng số mẫu đã xử lý thành công: {len(all_angle_errors_list)} / {len(df)}")
    print(f"Góc lệch trung bình (Average Angle Deviation): {mean_angle_error:.4f} độ")
    print("="*55)


if __name__ == "__main__":
    main()