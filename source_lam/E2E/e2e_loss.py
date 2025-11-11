# Tên file: e2e_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ps6d_model import square_distance

class E2ELoss(nn.Module):
    def __init__(self, weight_vote=1.0, weight_obj=1.0, weight_center=1.0, weight_orient=1.0,
                 match_threshold=0.05): # 5cm
        super().__init__()
        self.w_vote = weight_vote
        self.w_obj = weight_obj
        self.w_center = weight_center
        self.w_orient = weight_orient
        self.match_threshold = match_threshold # Ngưỡng MCE
        
        self.l1_loss = nn.SmoothL1Loss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') # Ổn định hơn
        self.cosine_loss = lambda pred, gt: (1.0 - torch.sum(pred * gt, dim=-1))

    def forward(self, preds, gts):
        # 1. Unpack đầu vào
        pred_votes = preds['pred_votes']           # (B, N, 3)
        pred_obj = preds['pred_objectness']      # (B, M)
        pred_center = preds['pred_center']       # (B, M, 3)
        pred_orient = preds['pred_orientation']  # (B, M, 3)
        
        scene_points, point_labels, gt_votes, gt_poses_list = gts
        B, N, _ = scene_points.shape
        M = self.num_proposals = pred_obj.shape[1]

        # 2. Loss Vote (Giống VoteNet)
        mask_fg = (point_labels > 0).float() # (B, N)
        loss_vote = self.l1_loss(pred_votes, gt_votes) * mask_fg.unsqueeze(-1)
        loss_vote = loss_vote.sum() / (mask_fg.sum() + 1e-6)

        # 3. Ghép cặp (Matching) và Tính toán Target
        # Tạo target rỗng
        gt_obj_labels = torch.zeros(B, M, dtype=torch.float32, device=pred_obj.device)
        gt_center_labels = torch.zeros(B, M, 3, dtype=torch.float32, device=pred_center.device)
        gt_orient_labels = torch.zeros(B, M, 3, dtype=torch.float32, device=pred_orient.device)
        
        loss_center_mask = torch.zeros(B, M, dtype=torch.float32, device=pred_center.device)
        loss_orient_mask = torch.zeros(B, M, dtype=torch.float32, device=pred_orient.device)

        for b in range(B):
            gt_poses = gt_poses_list[b] # (K, 6)
            if gt_poses.numel() == 0: continue # Không có GT

            gt_centers = gt_poses[:, 0:3].to(pred_center.device) # (K, 3)
            gt_orients = gt_poses[:, 3:6].to(pred_orient.device) # (K, 3)
            K = gt_centers.shape[0]

            # Tính khoảng cách giữa M dự đoán và K GT
            dist_matrix = square_distance(pred_center[b].unsqueeze(0), gt_centers.unsqueeze(0)).squeeze(0) # (M, K)
            
            # Tìm GT gần nhất cho mỗi proposal
            min_dist, min_idx = torch.min(dist_matrix, dim=1) # (M,)
            
            # Đánh dấu các proposal "dương tính" (gần 1 GT)
            positive_mask = (min_dist < self.match_threshold)
            
            gt_obj_labels[b, positive_mask] = 1.0
            
            # Gán GT cho các proposal dương tính
            assigned_gt_centers = gt_centers[min_idx[positive_mask]]
            assigned_gt_orients = gt_orients[min_idx[positive_mask]]
            
            gt_center_labels[b, positive_mask] = assigned_gt_centers
            gt_orient_labels[b, positive_mask] = assigned_gt_orients
            
            loss_center_mask[b, positive_mask] = 1.0
            loss_orient_mask[b, positive_mask] = 1.0
            
        # 4. Loss Objectness
        loss_obj = self.bce_loss(pred_obj, gt_obj_labels).mean()

        # 5. Loss Center (MCE)
        loss_center_l1 = self.l1_loss(pred_center, gt_center_labels) * loss_center_mask.unsqueeze(-1)
        loss_center = loss_center_l1.sum() / (loss_center_mask.sum() + 1e-6)

        # 6. Loss Orientation (OE)
        loss_orient_cos = self.cosine_loss(pred_orient, gt_orient_labels) * loss_orient_mask
        loss_orient = loss_orient_cos.sum() / (loss_orient_mask.sum() + 1e-6)

        # 7. Tổng Loss
        total_loss = (self.w_vote * loss_vote +
                      self.w_obj * loss_obj +
                      self.w_center * loss_center +
                      self.w_orient * loss_orient)
        
        return total_loss, {
            'L_total': total_loss.item(),
            'L_vote': loss_vote.item(),
            'L_obj': loss_obj.item(),
            'L_center': loss_center.item(),
            'L_orient': loss_orient.item()
        }