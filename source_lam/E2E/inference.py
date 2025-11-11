# Tên file: inference.py
import torch
import numpy as np
import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm

from e2e_model import VotePoseNet
try:
    from preprocess import get_point_cloud_from_mask, project_3d_to_2d
except ImportError:
    pass

# Hàm NMS (Non-Maximum Suppression) 3D đơn giản
def nms_3d(centers, scores, radius=0.05):
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        dist = np.linalg.norm(centers[i] - centers[indices[1:]], axis=1)
        inds = np.where(dist > radius)[0]
        indices = indices[inds + 1]
    return keep

def main_inference():
    # === THIẾT LẬP CẤU HÌNH ===
    MODEL_PATH = "/home/hp/VTAIRACE/source_lam/ps6d/e2e_model/e2e_model_best.pth"
    BASE_DATA_PATH = "/home/hp/VTAIRACE/source_lam/dataset/Train" 
    OUTPUT_CSV = "Submission3D_E2E.csv"
    
    # Dùng thông số từ file ảnh/code cũ
    color_intrinsics = {
        'fx': 643.90087890625, 'fy': 643.1365356445312,
        'cx': 650.2113037109375, 'cy': 355.79559326171875,
    }
    ROI_BOX = (560, 150, 300, 330) 
    u_min, v_min, w, h = ROI_BOX
    u_max = u_min + w
    v_max = v_min + h

    NUM_SCENE_POINTS = 20480
    NUM_PROPOSALS = 64
    FEATURE_DIM = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Tải Model (CHỈ 1 LẦN)
    model = VotePoseNet(
        feature_dim=FEATURE_DIM,
        num_proposals=NUM_PROPOSALS
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Đã tải model từ {MODEL_PATH}")

    rgb_files = sorted(glob.glob(os.path.join(BASE_DATA_PATH, "rgb", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(BASE_DATA_PATH, "depth", "*.png")))
    
    all_final_outputs = []

    for rgb_path, depth_path in tqdm(zip(rgb_files, depth_files), total=len(rgb_files)):
        IMAGE_FILENAME = os.path.basename(rgb_path)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue
            
        # 2. Tạo Point Cloud Toàn Cảnh (CHỈ 1 LẦN)
        # Tạo mask toàn bộ (hoặc chỉ ROI) để lấy PC
        full_mask = np.ones_like(depth, dtype=np.uint8) * 255
        scene_points = get_point_cloud_from_mask(full_mask, depth, color_intrinsics)
        
        if len(scene_points) < 100:
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue
            
        # Lấy mẫu
        if len(scene_points) > NUM_SCENE_POINTS:
            indices = np.random.choice(len(scene_points), NUM_SCENE_POINTS, replace=False)
        else:
            indices = np.random.choice(len(scene_points), NUM_SCENE_POINTS, replace=True)
        scene_points_sampled = scene_points[indices]
        
        points_tensor = torch.from_numpy(scene_points_sampled.astype(np.float32)).unsqueeze(0).to(device)
        
        # 3. Chạy Model (CHỈ 1 LẦN)
        with torch.no_grad():
            preds = model(points_tensor)

        pred_centers_np = preds['pred_center'][0].cpu().numpy()     # (M, 3)
        pred_orients_np = preds['pred_orientation'][0].cpu().numpy() # (M, 3)
        pred_scores_np = torch.sigmoid(preds['pred_objectness'][0]).cpu().numpy() # (M,)
        
        # 4. Lọc (Post-processing)
        # Lọc theo điểm tự tin
        SCORE_THRESH = 0.5
        score_mask = (pred_scores_np > SCORE_THRESH)
        if np.sum(score_mask) == 0:
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue
            
        pred_centers_filt = pred_centers_np[score_mask]
        pred_orients_filt = pred_orients_np[score_mask]
        pred_scores_filt = pred_scores_np[score_mask]

        # Lọc theo ROI
        roi_mask = np.zeros(len(pred_centers_filt), dtype=bool)
        for i, center in enumerate(pred_centers_filt):
            center_2d = project_3d_to_2d(center, color_intrinsics)
            if center_2d is not None:
                u_pred, v_pred = center_2d
                if (u_min <= u_pred < u_max and v_min <= v_pred < v_max):
                    roi_mask[i] = True
                    
        if np.sum(roi_mask) == 0:
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue
            
        pred_centers_roi = pred_centers_filt[roi_mask]
        pred_orients_roi = pred_orients_filt[roi_mask]
        pred_scores_roi = pred_scores_filt[roi_mask]
        
        # Lọc NMS
        keep_indices = nms_3d(pred_centers_roi, pred_scores_roi, radius=0.03) # 3cm
        
        candidate_results = []
        for i in keep_indices:
            candidate_results.append({
                'centroid': pred_centers_roi[i],
                'normal_vector': pred_orients_roi[i],
                'confidence': pred_scores_roi[i]
            })

        if not candidate_results:
            all_final_outputs.append((IMAGE_FILENAME, None, None, None, None, None, None))
            continue

        # 5. Lựa chọn cuối cùng 
        candidate_results.sort(key=lambda x: x['centroid'][2]) # Sắp xếp theo Z (gần nhất)
        selected = candidate_results[0] # ( có thể thêm logic gỡ hòa 'tie-break' ở đây)

        c = selected['centroid']
        n = selected['normal_vector']
        # Định dạng lại tên ảnh: thêm prefix "image_"
        BASE_NAME = os.path.splitext(IMAGE_FILENAME)[0]
        # Giữ nguyên số thứ tự, thêm "image_" phía trước
        formatted_name = f"image_{BASE_NAME.zfill(4)}.png" if BASE_NAME.isdigit() else f"image_{BASE_NAME}.png"

        all_final_outputs.append((formatted_name, c[0], c[1], c[2], n[0], n[1], n[2]))

    # Lưu kết quả
    df = pd.DataFrame(all_final_outputs, columns=['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print(f"Hoàn tất inference. Đã lưu vào {OUTPUT_CSV}")

if __name__ == "__main__":
    main_inference()