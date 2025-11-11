import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np
import glob

# TÁI SỬ DỤNG: Import hàm helper từ file preprocess.py 
from preprocess import get_point_cloud_from_mask, project_3d_to_2d

class E2ESceneDataset(Dataset):
    def __init__(self, base_path, mask_dir, gt_csv_path, num_scene_points, 
                 color_intrinsics, normalize=False):
        
        self.base_path = base_path
        self.mask_dir = mask_dir
        self.gt_df = pd.read_csv(gt_csv_path)
        self.color_intr = color_intrinsics
        self.num_scene_points = num_scene_points # Ví dụ: 20480
        self.normalize = normalize # Thường là False cho E2E

        self.image_filenames = sorted(self.gt_df['image_filename'].unique())
        self.gt_by_image = self.gt_df.groupby('image_filename')
        
        self.depth_dir = os.path.join(base_path, "depth")
        self.npy_mask_dir = os.path.join(self.mask_dir, "masks")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        try:
            # Lấy tên file GỐC từ CSV (ví dụ: 'image_0275.png')
            image_filename_from_csv = self.image_filenames[idx]
            
            # Lấy tên file THỰC TẾ trên ổ đĩa (ví dụ: '0275.png')
            actual_image_filename = image_filename_from_csv.replace("image_", "")
            
            # Lấy base_name THỰC TẾ (ví dụ: '0275')
            actual_base_name = os.path.splitext(actual_image_filename)[0]

            # 1. Lấy tất cả GT cho ảnh này (Dùng tên GỐC)
            gt_rows_for_image = self.gt_by_image.get_group(image_filename_from_csv)
            
            # Tải depth (Dùng tên THỰC TẾ)
            depth_path = os.path.join(self.depth_dir, actual_image_filename) # <--- SỬA Ở ĐÂY
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                # In ra lỗi rõ ràng hơn
                # print(f"LỖI: Không tìm thấy file {depth_path} (từ {image_filename_from_csv})")
                return None
            H, W = depth_img.shape[:2]

            # Tìm file mask (Dùng base_name THỰC TẾ)
            mask_files = sorted(glob.glob(os.path.join(self.npy_mask_dir, actual_base_name + "_*.npy"))) # <--- SỬA Ở ĐÂY

            scene_points_list = []
            point_labels_list = [] # 0 = nền, 1, 2, 3...
            point_votes_list = []
            gt_poses_list = [] # (K, 6)

            combined_mask = np.zeros((H, W), dtype=np.uint8)
            instance_id = 1

            # 2. Logic ghép cặp GT và Mask (Giữ nguyên)
            for _, gt_row in gt_rows_for_image.iterrows():
                gt_center_world = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
                gt_orient = np.array([gt_row['Rx'], gt_row['Ry'], gt_row['Rz']])
                
                gt_center_2d = project_3d_to_2d(gt_center_world, self.color_intr)
                if gt_center_2d is None: continue
                
                u_gt, v_gt = int(round(gt_center_2d[0])), int(round(gt_center_2d[1]))
                if not (0 <= v_gt < H and 0 <= u_gt < W): continue

                best_mask_img = None
                for mask_path in mask_files:
                    try:
                        mask_instance = np.load(mask_path)
                    except Exception as e:
                        # print(f"Lỗi đọc file mask: {mask_path} - {e}")
                        continue # Bỏ qua file mask hỏng
                        
                    mask_img = (mask_instance > 0.5).astype(np.uint8) * 255
                    if mask_img.shape[0] != H or mask_img.shape[1] != W:
                         mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    if mask_img[v_gt, u_gt] > 128:
                        best_mask_img = mask_img
                        mask_files.remove(mask_path) # Xóa mask đã dùng
                        break
                
                if best_mask_img is None: continue 

                # ... (Phần còn lại của hàm giữ nguyên) ...
                
            # 3. Tạo Point Cloud cho vật thể này
                obj_points = get_point_cloud_from_mask(best_mask_img, depth_img, self.color_intr)
                if len(obj_points) < 50: continue

                combined_mask = cv2.bitwise_or(combined_mask, best_mask_img)

                # 4. Tính Vote (lá phiếu)
                gt_votes = gt_center_world - obj_points
                
                scene_points_list.append(obj_points)
                point_labels_list.append(np.full(len(obj_points), instance_id))
                point_votes_list.append(gt_votes)
                gt_poses_list.append(np.hstack([gt_center_world, gt_orient]))
                
                instance_id += 1

            if not scene_points_list:
                return None # Không có vật thể nào hợp lệ

            # 5. Thêm điểm nền (Background)
            bg_mask = cv2.bitwise_not(combined_mask)
            # Lấy mẫu ngẫu nhiên điểm nền để tăng tốc
            bg_v, bg_u = np.where(bg_mask > 128)
            if len(bg_u) > 5000: # Giới hạn 5000 điểm nền
                bg_indices = np.random.choice(len(bg_u), 5000, replace=False)
                bg_v, bg_u = bg_v[bg_indices], bg_u[bg_indices]
            
            temp_bg_mask = np.zeros_like(bg_mask)
            temp_bg_mask[bg_v, bg_u] = 255
            
            bg_points = get_point_cloud_from_mask(temp_bg_mask, depth_img, self.color_intr)
            
            if len(bg_points) > 0:
                scene_points_list.append(bg_points)
                point_labels_list.append(np.full(len(bg_points), 0)) # Nhãn 0
                point_votes_list.append(np.zeros_like(bg_points)) # Vote 0

            # 6. Nối tất cả
            scene_points = np.concatenate(scene_points_list, axis=0)
            point_labels = np.concatenate(point_labels_list, axis=0)
            point_votes = np.concatenate(point_votes_list, axis=0)
            gt_poses = np.array(gt_poses_list).astype(np.float32)

            # 7. Lấy mẫu (Sample)
            if len(scene_points) > self.num_scene_points:
                indices = np.random.choice(len(scene_points), self.num_scene_points, replace=False)
            else:
                indices = np.random.choice(len(scene_points), self.num_scene_points, replace=True)

            scene_points = scene_points[indices]
            point_labels = point_labels[indices]
            point_votes = point_votes[indices]

            return (
                torch.from_numpy(scene_points.astype(np.float32)),
                torch.from_numpy(point_labels.astype(np.int64)),
                torch.from_numpy(point_votes.astype(np.float32)),
                torch.from_numpy(gt_poses)
            )
        except Exception as e:
            # print(f"Lỗi toàn cục khi xử lý {idx}: {e}")
            return None

def e2e_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    
    # Gộp các tensor
    scene_points, point_labels, point_votes, gt_poses = zip(*batch)
    
    scene_points = torch.stack(scene_points, 0)
    point_labels = torch.stack(point_labels, 0)
    point_votes = torch.stack(point_votes, 0)
    
    # gt_poses có kích thước K khác nhau, nên giữ nó ở dạng list
    return scene_points, point_labels, point_votes, gt_poses