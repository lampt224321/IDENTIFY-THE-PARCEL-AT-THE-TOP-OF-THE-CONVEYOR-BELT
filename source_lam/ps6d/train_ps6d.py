import torch
from ps6d_model import PS6DNetwork
from ps6d_dataset import ParcelDataset, train_ps6d
from torch.utils.data import DataLoader
import numpy as np
import os

# Camera intrinsics 
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

# Hàm collate_fn 
def collate_fn(batch):
    """
    Lọc ra các sample bị None (do lỗi load)
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

def main():
    base_path = r"/content/drive/MyDrive/ViettelAIRace/Train"
    # Dùng mask YOLACT để huấn luyện
    mask_dir = r"/content/drive/MyDrive/ViettelAIRace/yolact_result_train" 
    gt_csv_path = r"/content/drive/MyDrive/ViettelAIRace/Train/Public_train.csv"
    save_dir = '/content/drive/MyDrive/ViettelAIRace/ps6dmodel/conttrain'
    model_path = os.path.join(save_dir, 'ps6d_best.pth') 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = ParcelDataset(
        base_path=base_path,
        mask_dir=mask_dir,
        gt_csv_path=gt_csv_path,
        num_points=1024,
        color_intrinsics=color_intrinsics,
        depth_intrinsics=depth_intrinsics,
        R_d2c=R_depth_to_color,
        t_d2c=t_depth_to_color,
        normalize=True
    )

    print(f"Total samples: {len(dataset)}")

    # Chia Train/Val (Không đổi)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Tạo DataLoader (Không đổi)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ===========================================================
    # *** BẮT ĐẦU SỬA ĐỔI (LOGIC FINE-TUNE) ***
    # ===========================================================

    # 1. Khởi tạo model
    model = PS6DNetwork(num_points=1024, feature_dim=128)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # 2. Tải lại checkpoint tốt nhất (ps6d_best.pth) nếu có
    if os.path.exists(model_path):
        print(f"Đang tải lại model tốt nhất từ: {model_path} để fine-tune...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Tải thành công model có Val Loss: {checkpoint.get('loss', 'Không rõ')}")
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}. Huấn luyện từ đầu.")
    else:
        # Nếu bạn lỡ tay xóa file best.pth, nó sẽ tự huấn luyện từ đầu
        print("Không tìm thấy model checkpoint, huấn luyện từ đầu.")
        
    # 3. Gọi hàm huấn luyện với Learning Rate (LR) thấp hơn
    model = train_ps6d(
        model,
        train_loader,
        val_loader,
        num_epochs=100,      
        lr=0.0001,           
        device=device,
        save_dir=save_dir
    )
    # ===========================================================
    # *** KẾT THÚC SỬA ĐỔI ***
    # ===========================================================

    print("\n✅ Training hoàn tất!")

if __name__ == "__main__":
    main()