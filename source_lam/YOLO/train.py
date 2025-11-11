from ultralytics import YOLO
import os

def main():
    print("Bắt đầu quá trình huấn luyện (finetune)...")

    # 1. Tải mô hình gốc (pre-trained)
    model = YOLO('yolo11l.pt')

    # 2. Huấn luyện mô hình
    results = model.train(
        # --- Cấu hình cơ bản ---
        task='segment',
        data='my_dataset.yaml',
        epochs=300,
        imgsz=640,
        batch=32,

        project='runs/train',
        name='models',

        # --- Cấu hình Data Augmentation ---
        mosaic=1.0,       
        degrees=15.0,     
        translate=0.1,    
        scale=0.5,        
        
        fliplr=0.5,       
        flipud=0.0,       
        
        copy_paste=0.3,   

        # Biến đổi màu sắc & ánh sáng (Đã sửa)
        hsv_h=0.015,
        hsv_s=0.7,        # Kiểm soát độ bão hòa màu
        hsv_v=0.4,        # Kiểm soát độ sáng (Value/Brightness)
    )

    print("Hoàn tất huấn luyện!")

    save_directory = os.path.join(
        'runs/train',
        'models'
    )
    print(f"Trọng số tốt nhất đã được lưu tại: {save_directory}/weights/best.pt")

if __name__ == '__main__':
    main()

