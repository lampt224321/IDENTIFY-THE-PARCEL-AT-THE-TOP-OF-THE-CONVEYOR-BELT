from ultralytics import YOLO
import numpy as np
import os
import glob
from tqdm import tqdm  

# --- 1. CẤU HÌNH ---

MODEL_PATH = 'runs/train/models/weights/best.pt'

IMAGE_DIR = 'dataset/images/train/'

# Thư mục LƯU các file .npy
OUTPUT_DIR = 'runs/inference_npy/'


# ----------------------------

def run_inference_and_save_npy():
    print(f"Đang tải mô hình từ: {MODEL_PATH}")

    # 1. Tải mô hình ĐÃ HUẤN LUYỆN của bạn
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Lỗi: Không thể tải mô hình. Kiểm tra lại MODEL_PATH: {e}")
        return

    # 2. Tạo thư mục lưu .npy nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Lấy danh sách tất cả ảnh trong thư mục
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu suy luận...")

    # 4. Lặp qua từng ảnh, chạy suy luận và lưu
    #
    for img_path in tqdm(image_files, desc="Đang dự đoán mask"):
        try:
            # 4.1. Chạy dự đoán
            results = model(img_path, verbose=False) # verbose=False để tắt log

            # 4.2. Xây dựng đường dẫn lưu file
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_npy_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}.npy")

            # 4.3. Lấy dữ liệu mask
            if results and results[0].masks is not None:
                # Lấy dữ liệu mask và chuyển sang NumPy array
                # Shape sẽ là (N, H, W)
                # N = số vật thể, H, W = kích thước mask
                mask_data = results[0].masks.data.cpu().numpy()

                # 4.4. Lưu ra file .npy
                np.save(output_npy_path, mask_data)
            else:
                # Nếu mô hình không tìm thấy gì, lưu một mảng rỗng
                np.save(output_npy_path, np.array([]))

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

    print(f"\nHoàn tất suy luận! Đã lưu {len(image_files)} tệp .npy vào {OUTPUT_DIR}")

if __name__ == '__main__':
    run_inference_and_save_npy()
