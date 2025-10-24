import pandas as pd
import numpy as np
import os
import cv2 # Thư viện OpenCV để đọc ảnh
import re
from tqdm import tqdm
import math

# --- HẰNG SỐ CẤU HÌNH ---
GT_CSV_PATH = "/Users/angelinacu/Desktop/Study/Viettel/ThiSinh/Train/Public train.csv"
YOLO_RESULT_DIR = "/Users/angelinacu/Desktop/Study/Viettel/IDENTIFY-THE-PARCEL-AT-THE-TOP-OF-THE-CONVEYOR-BELT/source_haanh/yolo_result"
DEPTH_IMG_DIR = "/Users/angelinacu/Desktop/Study/Viettel/ThiSinh/Train/depth"
OUTPUT_CSV_PATH = "predicted_raw_vs_gt_coordinates.csv" # Giữ tên file này

# --- Ngưỡng ---
HEIGHT_TOLERANCE_M = 0.005 # 5mm
# --- [THÊM LẠI] ---
CORRECTNESS_THRESHOLD_M = 0.01 # 10mm (1cm) - Ngưỡng để coi là dự đoán đúng
# --- [KẾT THÚC THÊM LẠI] ---

# --- THAM SỐ CAMERA (Màu - Color) ---
color_intr = {
    "fx": 643.90087890625,
    "fy": 643.1365356445312,
    "cx": 650.2113037109375,
    "cy": 355.79559326171875,
    "coeffs": [-0.05658450722694397, 0.06544225662946701, -0.0008694113348610699, 0.00016751799557823688, -0.020957745611667633]
}
color_camera_matrix = np.array([
    [color_intr["fx"], 0, color_intr["cx"]],
    [0, color_intr["fy"], color_intr["cy"]],
    [0, 0, 1]
])
color_dist_coeffs = np.array(color_intr["coeffs"])

# --- THAM SỐ CAMERA (Độ sâu - Depth) ---
depth_intr = {
    "fx": 650.0616455078125,
    "fy": 650.0616455078125,
    "cx": 649.5928955078125,
    "cy": 360.9415588378906
}

def get_image_filename_from_txt(txt_filename):
    """Chuyển '0000.txt' thành 'image_0000.png'"""
    base_name = os.path.splitext(txt_filename)[0]
    return f"image_{base_name}.png"

def calculate_3d_point(u, v, depth_value_mm, intrinsics):
    """
    Chiếu ngược điểm ảnh 2D + độ sâu sang 3D (hệ tọa độ camera depth).
    """
    if depth_value_mm == 0:
        return None
    z_d = float(depth_value_mm) / 1000.0 # Chuyển mm sang m
    x_d = (u - intrinsics["cx"]) * z_d / intrinsics["fx"]
    y_d = (v - intrinsics["cy"]) * z_d / intrinsics["fy"]
    return np.array([x_d, y_d, z_d]), (u, v)

def main():
    # 1. Đọc Ground Truth
    try:
        gt_df = pd.read_csv(GT_CSV_PATH)
        gt_df.set_index('image_filename', inplace=True)
        print(f"Đọc thành công {len(gt_df)} dòng ground truth.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file Ground Truth CSV tại: {GT_CSV_PATH}")
        return
    except KeyError:
        print("LỖI: File CSV không có cột 'image_filename'.")
        return

    results_list = []
    # --- [THÊM LẠI] ---
    all_errors = [] # Danh sách lưu lỗi Euclid (mét)
    correct_predictions_count = 0 # Đếm số dự đoán đúng
    # --- [KẾT THÚC THÊM LẠI] ---

    yolo_files = [f for f in os.listdir(YOLO_RESULT_DIR) if f.endswith('.txt')]
    print(f"Tìm thấy {len(yolo_files)} file kết quả YOLO.")

    # 2. Quét file YOLO
    for yolo_filename in tqdm(yolo_files, desc="Processing images"):
        base_name = os.path.splitext(yolo_filename)[0]
        image_filename_gt = get_image_filename_from_txt(yolo_filename)

        if image_filename_gt not in gt_df.index:
            continue

        yolo_filepath = os.path.join(YOLO_RESULT_DIR, yolo_filename)
        depth_filepath = os.path.join(DEPTH_IMG_DIR, f"{base_name}.png")

        if not os.path.exists(depth_filepath):
            continue

        # 3. Xử lý từng ảnh
        detected_packets_info = []

        try:
            depth_image = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
            if depth_image is None: continue

            with open(yolo_filepath, 'r') as f: lines = f.readlines()
            if not lines: continue

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 8 or parts[0] != 'packet': continue

                try:
                    center_x_rgb_distorted = float(parts[6])
                    center_y_rgb_distorted = float(parts[7])
                    distorted_point = np.array([[[center_x_rgb_distorted, center_y_rgb_distorted]]], dtype=np.float32)

                    undistorted_normalized = cv2.undistortPoints(distorted_point, color_camera_matrix, color_dist_coeffs, P=color_camera_matrix)
                    u_undistorted = undistorted_normalized[0][0][0]
                    v_undistorted = undistorted_normalized[0][0][1]
                    u_lookup = int(round(u_undistorted))
                    v_lookup = int(round(v_undistorted))

                    if 0 <= v_lookup < depth_image.shape[0] and 0 <= u_lookup < depth_image.shape[1]:
                        depth_value_mm = depth_image[v_lookup, u_lookup]
                        result = calculate_3d_point(u_lookup, v_lookup, depth_value_mm, depth_intr)
                        if result is not None:
                            point_3d_depth_cam, original_uv = result
                            detected_packets_info.append((point_3d_depth_cam, original_uv))
                except ValueError: continue
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_filename_gt}: {e}")
            continue

        # 4. Tìm vật thể trên cùng (LOGIC MỚI)
        if not detected_packets_info: continue

        min_z = min(info[0][2] for info in detected_packets_info)
        top_level_packets_info = [
            info for info in detected_packets_info
            if info[0][2] <= min_z + HEIGHT_TOLERANCE_M
        ]

        if len(top_level_packets_info) == 1:
            topmost_packet_info = top_level_packets_info[0]
        else:
            topmost_packet_info = max(top_level_packets_info, key=lambda info: info[0][2])

        topmost_packet_3d_depth_cam = topmost_packet_info[0]

        # 5. Lấy Ground Truth, Lưu CSV và Tính Lỗi
        try:
            gt_row = gt_df.loc[image_filename_gt]
            gt_point = np.array([gt_row['x'], gt_row['y'], gt_row['z']])

            # Tọa độ dự đoán RAW (hệ camera depth)
            predicted_point_raw = topmost_packet_3d_depth_cam

            # Áp dụng Z-flip *chỉ để tính lỗi*
            predicted_point_transformed_for_error = np.array([
                predicted_point_raw[0],
                -predicted_point_raw[1],
                -predicted_point_raw[2]
            ])
            # Tính lỗi Euclid (mét)
            error = np.linalg.norm(predicted_point_transformed_for_error - gt_point)
            all_errors.append(error) # <<-- Thêm lỗi vào danh sách

            # Kiểm tra độ chính xác
            if error <= CORRECTNESS_THRESHOLD_M:
                correct_predictions_count += 1
            # --- [KẾT THÚC BỔ SUNG LẠI TÍNH LỖI] ---

            # Thêm tọa độ RAW vào danh sách kết quả CSV
            results_list.append({
                'pred_x': predicted_point_raw[0], # Dấu gốc
                'pred_y': predicted_point_raw[1], # Dấu gốc
                'pred_z': predicted_point_raw[2], # Dấu gốc
                'gt_x': gt_point[0],
                'gt_y': gt_point[1],
                'gt_z': gt_point[2]
            })

        except KeyError: pass
        except Exception as e:
            print(f"Lỗi khi xử lý ground truth cho {image_filename_gt}: {e}")

    # --- KẾT THÚC VÒNG LẶP ---

    # 6. Lưu kết quả CSV
    if not results_list:
        print("\nKhông có kết quả nào để lưu.")
        # Nếu không có kết quả lưu CSV, cũng không thể tính MCE/Accuracy
        print("\nKhông tính được MCE và Accuracy.")
        return # Thoát sớm
    else:
        results_df = pd.DataFrame(results_list)
        try:
            results_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
            print(f"\nĐã lưu {len(results_df)} cặp tọa độ (dấu gốc) vào: {OUTPUT_CSV_PATH}")
        except Exception as e:
            print(f"\nLỖI khi lưu file CSV: {e}")

    # --- [SỬA LẠI PHẦN CUỐI] ---
    # 7. Tính và In MCE (mm, capped) và Accuracy (%)
    if not all_errors:
        print("\nKhông tính được MCE và Accuracy (Không có lỗi nào được ghi lại).")
        return

    # A. Tính MCE theo công thức mới (capped error)
    capped_errors = [min(error, 0.05) for error in all_errors]
    mean_center_error_m_capped = np.mean(capped_errors)
    mean_center_error_mm_capped = mean_center_error_m_capped * 1000.0

    # B. Tính Accuracy
    total_predictions_compared = len(all_errors)
    # Đếm số dự đoán đúng dựa trên lỗi GỐC (chưa bị cap)
    # Lưu ý: correct_predictions_count đã được tính trong vòng lặp
    accuracy_percentage = (correct_predictions_count / total_predictions_compared) * 100.0 if total_predictions_compared > 0 else 0.0

    # C. In kết quả
    print("\n" + "="*60)
    print("--- TỔNG KẾT TÍNH MCE (Capped) & ACCURACY ---")
    print(f"Số ảnh đã xử lý và so sánh thành công: {total_predictions_compared}")
    print(f"Sai số tọa độ trung tâm (MCE - Capped @ 0.05m): {mean_center_error_mm_capped:.4f} mm")
    print(f"Độ chính xác (Accuracy @ {CORRECTNESS_THRESHOLD_M*1000:.0f}mm): {accuracy_percentage:.2f} %")
    print("="*60)
    # --- [KẾT THÚC SỬA LẠI PHẦN CUỐI] ---

if __name__ == "__main__":
    main()