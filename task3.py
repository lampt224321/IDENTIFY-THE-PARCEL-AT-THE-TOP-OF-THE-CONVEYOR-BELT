import os
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# BƯỚC 1: THIẾT LẬP CÁC THAM SỐ CỐ ĐỊNH
# ==============================================================================

# --- Tham số nội tại camera màu ---
COLOR_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
    width=1280, height=720,
    fx=643.90087890625, fy=643.1365356445312,
    cx=650.2113037109375, cy=355.79559326171875
)

# --- Vùng quan tâm (ROI) ---
ROI = (560, 150, 300, 330) # (x, y, w, h)

# --- Các tham số cho thuật toán ---
DEPTH_SCALE = 1000.0 # Giả định: giá trị pixel trong ảnh depth là mm
PLANE_DISTANCE_THRESHOLD = 0.01 # 1cm
DBSCAN_EPS = 0.02 # 2cm
DBSCAN_MIN_POINTS = 10
HEIGHT_TOLERANCE = 0.005 # 5mm


# ==============================================================================
# BƯỚC 2: HÀM XỬ LÝ CHO MỘT CẶP ẢNH
# ==============================================================================

def find_topmost_parcel(color_path, depth_path):
    """
    Hàm này xử lý một cặp ảnh màu và độ sâu, trả về thông tin của bưu kiện trên cùng.
    Trả về một dictionary chứa kết quả hoặc None nếu không tìm thấy.
    """
    try:
        # --- Đọc và Căn chỉnh dữ liệu ---
        color_raw = o3d.io.read_image(color_path)
        depth_raw = o3d.io.read_image(depth_path)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=DEPTH_SCALE, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, COLOR_INTRINSICS)

        # --- Phân vùng Vật thể ---
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=PLANE_DISTANCE_THRESHOLD, ransac_n=3, num_iterations=1000
        )
        parcels_pcd = pcd.select_by_index(inlier_indices, invert=True)
        
        if not parcels_pcd.has_points():
            return None # Không có điểm nào sau khi loại bỏ mặt phẳng

        # --- Phân cụm để tách riêng từng bưu kiện ---
        labels = np.array(parcels_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
        
        individual_parcels = {}
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        for label_id in unique_labels:
            indices = np.where(labels == label_id)[0]
            individual_parcels[label_id] = parcels_pcd.select_by_index(indices)

        # --- Xác định Bưu kiện Ứng viên trong ROI ---
        candidate_parcels = []
        roi_x, roi_y, roi_w, roi_h = ROI
        for label_id, pcd_item in individual_parcels.items():
            center_3d = pcd_item.get_center()
            X, Y, Z = center_3d
            if Z == 0: continue
            u = int((X * COLOR_INTRINSICS.intrinsic_matrix[0, 0] / Z) + COLOR_INTRINSICS.intrinsic_matrix[0, 2])
            v = int((Y * COLOR_INTRINSICS.intrinsic_matrix[1, 1] / Z) + COLOR_INTRINSICS.intrinsic_matrix[1, 2])
            
            if (roi_x <= u < roi_x + roi_w) and (roi_y <= v < roi_y + roi_h):
                candidate_parcels.append({"label": label_id, "pcd": pcd_item, "center_3d": center_3d})

        if not candidate_parcels:
            return None # Không có bưu kiện nào trong ROI

        # --- Chọn Bưu kiện Mục tiêu ---
        candidate_parcels.sort(key=lambda item: item['center_3d'][2])
        highest_parcel = candidate_parcels[0]
        top_group = [p for p in candidate_parcels if p['center_3d'][2] - highest_parcel['center_3d'][2] < HEIGHT_TOLERANCE]
        
        if len(top_group) > 1:
            top_group.sort(key=lambda item: item['center_3d'][1], reverse=True)
        
        target_parcel = top_group[0]

        # --- Tính toán Tọa độ 3D và Hướng gắp ---
        center_3d = target_parcel['center_3d']
        points = np.asarray(target_parcel['pcd'].points)
        
        covariance_matrix = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
        
        if np.dot(normal_vector, -center_3d) > 0:
            normal_vector = -normal_vector
            
        rotation_matrix, _ = R.align_vectors([[0, 0, 1]], [normal_vector])
        Rx, Ry, Rz = rotation_matrix.as_euler('xyz', degrees=True)

        return {
            "x": center_3d[0], "y": center_3d[1], "z": center_3d[2],
            "Rx": Rx, "Ry": Ry, "Rz": Rz
        }

    except Exception as e:
        print(f"Lỗi khi xử lý {os.path.basename(color_path)}: {e}")
        return None

# ==============================================================================
# BƯỚC 3: VÒNG LẶP CHÍNH VÀ XUẤT KẾT QUẢ
# ==============================================================================

def main():
    # --- Định nghĩa đường dẫn (thay đổi cho phù hợp) ---
    data_dir = "./dataset/train" # Thư mục chứa các thư mục rgb, depth
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")

    # --- Lấy danh sách file ảnh và sắp xếp để đảm bảo đúng thứ tự ---
    image_filenames = sorted(os.listdir(rgb_dir))
    
    # --- Danh sách để lưu tất cả kết quả ---
    all_results = []

    print("🚀 Bắt đầu xử lý bộ dữ liệu...")
    for filename in image_filenames:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            color_path = os.path.join(rgb_dir, filename)
            # Giả định file depth có cùng tên
            depth_path = os.path.join(depth_dir, filename) 

            if not os.path.exists(depth_path):
                print(f" Không tìm thấy file depth tương ứng cho {filename}")
                continue

            print(f"   -> Đang xử lý: {filename}")
            result = find_topmost_parcel(color_path, depth_path)

            if result:
                result_row = {
                    "image_filename": filename,
                    "x": result["x"], "y": result["y"], "z": result["z"],
                    "Rx": result["Rx"], "Ry": result["Ry"], "Rz": result["Rz"]
                }
                all_results.append(result_row)
            else:
                 print(f"-> Không tìm thấy bưu kiện hợp lệ trong {filename}")


    # --- Xuất kết quả ra file CSV ---
    if all_results:
        submission_df = pd.DataFrame(all_results)
        # Sắp xếp lại cột theo đúng thứ tự yêu cầu
        submission_df = submission_df[["image_filename", "x", "y", "z", "Rx", "Ry", "Rz"]]
        output_path = "Submission_3D.csv"
        submission_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nHoàn thành! Đã lưu kết quả vào file {output_path}")
    else:
        print("\n Không có kết quả nào để ghi.")

if __name__ == "__main__":
    main()