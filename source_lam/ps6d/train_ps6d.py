import torch
from ps6d_model import PS6DNetwork, PS6DLoss
from ps6d_dataset import PreprocessedDataset, train_ps6d
from torch.utils.data import DataLoader
import numpy as np
import os

# Hàm collate_fn
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)
# ===========================================================
# HÀM MAIN
# ===========================================================
def main():
    PREPROCESSED_DIR = "/home/hp/VTAIRACE/source_lam/ps6d/preprocessed_data"
    save_dir = '/home/hp/VTAIRACE/source_lam/ps6d/ps6dmodel/testmodel'
    model_path = os.path.join(save_dir, 'ps6d_best.pth')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = PreprocessedDataset(data_dir=PREPROCESSED_DIR)

    print(f"Total *preprocessed* samples: {len(dataset)}")

    # Chia Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    try:
        cpu_count = os.cpu_count()
        num_workers = max(1, cpu_count // 2)
        print(f"Phát hiện {cpu_count} CPU, sử dụng {num_workers} workers.")
    except:
        num_workers = 8
        print(f"Không thể tự động phát hiện CPU, đặt num_workers={num_workers}")

    BATCH_SIZE = 8 

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # 1. Khởi tạo model
    model = PS6DNetwork(num_points=2048, feature_dim=128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    print("Huấn luyện model (MCE + OE 90/10) từ đầu (trên dữ liệu đã tiền xử lý)...")

    NUM_EPOCHS_TO_RUN = 500

    model, history = train_ps6d(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS_TO_RUN,
        lr=0.001,
        device=device,
        save_dir=save_dir
    )

    print("\n✅ Training hoàn tất!")

    history_save_path = os.path.join(save_dir, 'training_history.npy')
    np.save(history_save_path, history)
    print(f"Đã lưu lịch sử training (history) tại: {history_save_path}")

if __name__ == "__main__":
    main()