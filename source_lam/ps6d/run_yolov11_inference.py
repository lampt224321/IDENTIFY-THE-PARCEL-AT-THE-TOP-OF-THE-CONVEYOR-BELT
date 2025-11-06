import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

print("Bắt đầu chạy inference với model YOLO...")

# ===================================================================
# BƯỚC 1: CẤU HÌNH (BẠN CẦN CHỈNH SỬA Ở ĐÂY)
# ===================================================================

# --- 1. Chỉ định đường dẫn đến model YOLO đã finetune của bạn ---
YOLO_MODEL_PATH = "/home/hp/VTAIRACE/Code_Angie/yolov11_results/best.pt" # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY

# --- 2. Cấu hình đường dẫn Input / Output ---
IMAGE_DIR = "/home/hp/VTAIRACE/source_lam/dataset/Train/rgb"
OUTPUT_DIR_BASE = "/home/hp/VTAIRACE/source_lam/yolov11_results" 

# --- 3. Cấu hình Model ---
CONF_THRESHOLD = 0.3  
IOU_THRESHOLD = 0.5   


# ===================================================================
# PHẦN CÒN LẠI CỦA SCRIPT (Không cần chỉnh sửa)
# ===================================================================

# Tạo thư mục output nếu chưa có
OUTPUT_MASK_DIR = os.path.join(OUTPUT_DIR_BASE, "masks")
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
print(f"Sẽ lưu các file .npy vào: {OUTPUT_MASK_DIR}")

# --- Tải model YOLO ---
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"Tải model từ {YOLO_MODEL_PATH} thành công.")
except Exception as e:
    print(f"LỖI: Không thể tải model tại {YOLO_MODEL_PATH}.")
    print(f"Chi tiết lỗi: {e}")
    exit()

# Lấy kích thước ảnh chuẩn
IMAGE_HEIGHT, IMAGE_WIDTH = 720, 1280 

# --- Lặp qua tất cả ảnh ---
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
if not image_files:
    print(f"LỖI: Không tìm thấy ảnh .png nào trong {IMAGE_DIR}")
    exit()

print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")

for img_path in tqdm(image_files, desc="Đang tạo mask"):
    BASE_NAME = os.path.splitext(os.path.basename(img_path))[0]
    
    # --- Chạy model predict ---
    try:
        # === THAY ĐỔI Ở ĐÂY ===
        # Đã xóa 'imgsz=IMAGE_SIZE_PREDICT'
        results = model.predict(
            img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        # ====================
    except Exception as e:
        print(f"Lỗi khi predict ảnh {BASE_NAME}: {e}")
        continue

    result = results[0] 
    
    if result.masks is None:
        continue

    # Lấy dữ liệu mask và class ID
    masks_data = result.masks.data
    
    try:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        print(f"Lỗi khi lấy class_ids cho {BASE_NAME}: {e}. Bỏ qua ảnh này.")
        continue

    # Kiểm tra số lượng mask và class ID có khớp không
    if len(masks_data) != len(class_ids):
        print(f"Cảnh báo: Số lượng mask ({len(masks_data)}) và class ID ({len(class_ids)}) không khớp cho ảnh {BASE_NAME}.")
        num_detections = min(len(masks_data), len(class_ids))
    else:
        num_detections = len(masks_data)

    # --- Lưu từng mask ra file .npy ---
    num_saved = 0
    for i in range(num_detections):
        # Lấy mask
        mask_tensor = masks_data[i]
        mask_numpy = mask_tensor.cpu().numpy()
        
        # Lấy class ID
        class_id = class_ids[i]
        
        # Resize mask về kích thước ảnh gốc
        full_size_mask = cv2.resize(
            mask_numpy, 
            (IMAGE_WIDTH, IMAGE_HEIGHT), 
            interpolation=cv2.INTER_NEAREST
        )
        
        final_mask_data = (full_size_mask > 0.5) # Lưu dưới dạng Boolean
        
        # Tên file mới:
        output_filename = f"{BASE_NAME}_{i}_class{class_id}.npy"
        
        output_path = os.path.join(OUTPUT_MASK_DIR, output_filename)
        
        # Lưu file
        np.save(output_path, final_mask_data)
        num_saved += 1

print(f"\n✅ Hoàn tất quá trình tạo {len(image_files)} ảnh.")
print(f"Toàn bộ mask đã được lưu tại: {OUTPUT_MASK_DIR}")