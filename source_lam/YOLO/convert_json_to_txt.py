import json
import os
import glob
from tqdm import tqdm

# --- 1. CẤU HÌNH ---

# R Ánh xạ (mapping) từ tên label sang ID
# ID này PHẢI KHỚP với file my_dataset.yaml
CLASS_MAPPING = {
    "pack": 0
}

# Đường dẫn đến thư mục chứa các file .json
JSON_DIR = "/content/drive/MyDrive/ViettelAIRace/YOLO/Train/label"

TXT_OUTPUT_DIR = "/content/drive/MyDrive/ViettelAIRace/YOLO/Train/label/train"

# ---------------------------------

def convert_labelme_to_yolo():
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

    if not json_files:
        print(f"Lỗi: Không tìm thấy file .json nào trong {JSON_DIR}")
        return

    print(f"Đang chuyển đổi {len(json_files)} file JSON sang định dạng YOLO .txt...")

    for json_path in tqdm(json_files, desc="Chuyển đổi JSON"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Lấy kích thước ảnh từ file JSON (rất quan trọng để chuẩn hóa)
            img_height = data['imageHeight']
            img_width = data['imageWidth']

            if img_height <= 0 or img_width <= 0:
                print(f"Bỏ qua {json_path}: Kích thước ảnh không hợp lệ.")
                continue

            yolo_lines = []

            # Lặp qua tất cả các "shapes" (vật thể) trong file JSON
            for shape in data.get('shapes', []):
                label_name = shape.get('label')

                # Kiểm tra xem label này có trong mapping không
                if label_name in CLASS_MAPPING:
                    class_id = CLASS_MAPPING[label_name]

                    # Lấy danh sách các điểm [x, y]
                    points = shape.get('points', [])
                    if not points or shape.get('shape_type') != 'polygon':
                        continue # Bỏ qua nếu không phải polygon hoặc không có điểm

                    # Chuyển đổi và chuẩn hóa các điểm
                    normalized_points = []
                    for x, y in points:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        # Giới hạn giá trị trong khoảng [0, 1]
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        normalized_points.append(f"{norm_x:.6f}") # Thêm 6 chữ số thập phân
                        normalized_points.append(f"{norm_y:.6f}")

                    # Tạo dòng YOLO: <class_id> x1 y1 x2 y2 ...
                    line = f"{class_id} " + " ".join(normalized_points)
                    yolo_lines.append(line)

            # Ghi tất cả các dòng cho ảnh này ra file .txt
            if yolo_lines:
                # Lấy tên file (ví dụ: image1.json -> image1.txt)
                base_name = os.path.basename(json_path)
                txt_name = os.path.splitext(base_name)[0] + ".txt"
                txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)

                with open(txt_path, 'w') as f_txt:
                    f_txt.write("\n".join(yolo_lines))

        except Exception as e:
            print(f"\nLỗi khi xử lý file {json_path}: {e}")

    print("\nHoàn tất chuyển đổi!")
    print(f"Đã lưu các file .txt vào: {TXT_OUTPUT_DIR}")

if __name__ == '__main__':
    convert_labelme_to_yolo()
