# Tên file: train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Import các file mới
from e2e_dataset import E2ESceneDataset, e2e_collate_fn
from e2e_model import VotePoseNet
from e2e_loss import E2ELoss

def main():
    # === THIẾT LẬP CẤU HÌNH ===
    BASE_DATA_PATH = "/home/hp/VTAIRACE/source_lam/dataset/Train"
    MASK_DIR = "/home/hp/VTAIRACE/Code_Angie/masks_npy"
    GT_CSV = "/home/hp/VTAIRACE/source_lam/dataset/Train/Public_train.csv"
    SAVE_DIR = "/home/hp/VTAIRACE/source_lam/E2E/models"
    
    color_intrinsics = {
        'fx': 643.90087890625, 'fy': 643.1365356445312,
        'cx': 650.2113037109375, 'cy': 355.79559326171875,
    }
    
    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    LR = 0.001
    NUM_SCENE_POINTS = 20480
    NUM_PROPOSALS = 64
    FEATURE_DIM = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # 1. Dataset
    dataset = E2ESceneDataset(
        base_path=BASE_DATA_PATH,
        mask_dir=MASK_DIR,
        gt_csv_path=GT_CSV,
        num_scene_points=NUM_SCENE_POINTS,
        color_intrinsics=color_intrinsics
    )
    # (Bạn có thể chia train/val ở đây)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=e2e_collate_fn # Dùng collate_fn mới
    )

    # 2. Model
    model = VotePoseNet(
        feature_dim=FEATURE_DIM,
        num_proposals=NUM_PROPOSALS
    ).to(device)

    # 3. Loss & Optimizer
    criterion = E2ELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🚀 Bắt đầu training mô hình E2E (VotePoseNet)...")

    # 4. Training Loop (Tương tự file cũ của bạn)
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            if batch is None: continue
            
            scene_points, point_labels, point_votes, gt_poses_list = batch
            
            # Gửi tensor lên GPU
            scene_points = scene_points.to(device)
            point_labels = point_labels.to(device)
            point_votes = point_votes.to(device)
            # gt_poses_list giữ nguyên là list
            
            gts = (scene_points, point_labels, point_votes, gt_poses_list)
            
            # Forward
            preds = model(scene_points)
            
            # Loss
            loss, loss_dict = criterion(preds, gts)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # (Thêm logic Val và Save Model ở đây)
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(SAVE_DIR, f'e2e_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Đã lưu model tại {save_path}")

if __name__ == "__main__":
    main()