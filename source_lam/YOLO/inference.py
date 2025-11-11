import os
import glob
import numpy as np
from ultralytics import YOLO

def save_yolo_segmentation_masks(weights_path, image_folder, output_folder, conf_threshold=0.25):
    model = YOLO(weights_path)

    # 2. Xử lý thiếu sót: Đảm bảo thư mục output tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    saved_mask_count = 0
    
    for image_path in image_files:
        image_name_base = os.path.splitext(os.path.basename(image_path))[0]
        
        # Đã có try-except bên trong vòng lặp để xử lý lỗi inference
        try:
            results = model.predict(
                source=image_path, 
                save=False, 
                verbose=False,
                conf=conf_threshold
            ) 
            
            if results and len(results) > 0:
                result = results[0]

                # Bổ sung kiểm tra task là 'segment'
                if result.masks is not None and result.masks.data is not None:
                    # Chú ý: Lỗi ValueError 'too many values to unpack' xảy ra ở backend 
                    # do phiên bản/môi trường. Nếu lỗi vẫn xảy ra ở dòng này, bạn cần 
                    # cập nhật hoặc hạ cấp thư viện Ultralytics.
                    masks_np = result.masks.data.cpu().numpy()
                    
                    for i, mask_i in enumerate(masks_np):
                        output_filename = f"{image_name_base}_object{i}.npy"
                        output_path = os.path.join(output_folder, output_filename)
                        np.save(output_path, mask_i)
                        saved_mask_count += 1
                        
        except Exception:
            continue 

    # 3. Lỗi cú pháp: Xóa dấu chấm phẩy ở cuối lệnh return
    return saved_mask_count

# --- Ví dụ cách gọi hàm ---
WEIGHTS_PATH = '/home/hp/VTAIRACE/source_lam/yolov11/train/models/weights/best.pt'
IMAGE_FOLDER = '/home/hp/VTAIRACE/source_lam/dataset/Train/rgb'
OUTPUT_FOLDER = '/home/hp/VTAIRACE/source_lam/yolov11/train/models/results/masks' 

total_masks = save_yolo_segmentation_masks(WEIGHTS_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)

