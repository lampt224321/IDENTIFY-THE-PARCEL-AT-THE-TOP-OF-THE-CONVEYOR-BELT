# Tên file: e2e_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# TÁI SỬ DỤNG: Import các khối từ file ps6d_model.py 
try:
    from ps6d_model import (
        SetAbstraction, FeaturePropagation, PointTransformerLayer, 
        farthest_point_sample, query_ball_point, square_distance
    )
except ImportError:
    print("Vui lòng đảm bảo file ps6d_model.py ở cùng thư mục")
    pass

class VotePoseNet(nn.Module):
    def __init__(self, feature_dim=128, num_proposals=64, 
                 group_radius=0.1, num_group_points=32):
        super().__init__()
        self.num_proposals = num_proposals
        self.group_radius = group_radius
        self.num_group_points = num_group_points

        # 1. Backbone (Bộ trích xuất đặc trưng)
        # TÁI SỬ DỤNG: Dùng y hệt backbone của PS6DNetwork
        self.sa1 = SetAbstraction(512, 0.05, 32, 3, [32, 32, 64])
        self.pt1 = PointTransformerLayer(64, 64)
        self.sa2 = SetAbstraction(128, 0.1, 64, 64 + 3, [64, 64, 128])
        self.pt2 = PointTransformerLayer(128, 128)
        self.sa3 = SetAbstraction(None, None, None, 128 + 3, [128, 128, 256])
        self.fp3 = FeaturePropagation(384, [256, 128])
        self.fp2 = FeaturePropagation(192, [128, 64])
        self.fp1 = FeaturePropagation(67, [64, 64, feature_dim])

        # 2. Voting Head (Module Bỏ phiếu)
        # Thay thế centroid_head và rotation_head cũ
        self.vote_head = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 3, 1) # Chỉ dự đoán 3D offset
        )
        
        # 3. Proposal Module (Module Đề xuất)
        # Một mini-PointNet để xử lý các điểm đã gom nhóm
        self.proposal_aggregator = nn.Sequential(
            nn.Conv2d(feature_dim + 3, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1)
        )
        
        # Các đầu ra cuối cùng
        self.proposal_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # 1 (Objectness) + 3 (Center residual) + 3 (Orientation vector)
            nn.Conv1d(128, 1 + 3 + 3, 1) 
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # 1. Chạy Backbone (Y hệt PS6DNetwork)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l1_points = self.pt1(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.pt2(l2_points.permute(0, 2, 1)).permute(0, 2, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points_feat = self.fp1(xyz, l1_xyz, xyz, l1_points) # (B, N, feature_dim)

        feat_trans = l0_points_feat.permute(0, 2, 1) # (B, feature_dim, N)
        
        # 2. Dự đoán Vote (Module Bỏ phiếu)
        pred_votes = self.vote_head(feat_trans).permute(0, 2, 1) # (B, N, 3)
        voted_xyz = xyz + pred_votes # (B, N, 3)

        # 3. Lấy mẫu + Gom nhóm (Module Đề xuất)
        # Dùng FPS trên các điểm đã vote để chọn ra M "hạt giống"
        seed_fps_indices = farthest_point_sample(voted_xyz, self.num_proposals) # (B, M)
        
        # Lấy M điểm hạt giống
        seed_xyz = torch.gather(voted_xyz, 1, seed_fps_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, M, 3)
        
        # Gom đặc trưng (feature) xung quanh các hạt giống
        # Chúng ta gom từ CẢ xyz GỐC và CÁC ĐẶC TRƯNG GỐC
        group_indices = query_ball_point(self.group_radius, self.num_group_points, xyz, seed_xyz)
        
        # grouped_xyz: (B, M, K, 3)
        grouped_xyz = torch.gather(xyz.unsqueeze(1).expand(-1, self.num_proposals, -1, -1), 
                                   2, group_indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        # Chuẩn hóa
        grouped_xyz_norm = grouped_xyz - seed_xyz.unsqueeze(2)

        # grouped_features: (B, M, K, feature_dim)
        grouped_features = torch.gather(l0_points_feat.unsqueeze(1).expand(-1, self.num_proposals, -1, -1),
                                        2, group_indices.unsqueeze(-1).expand(-1, -1, -1, l0_points_feat.shape[-1]))
        
        # Gộp đặc trưng hình học và đặc trưng học được
        grouped_input = torch.cat([grouped_xyz_norm, grouped_features], dim=-1) # (B, M, K, 3+feature_dim)
        
        # 4. Xử lý (Aggregation)
        grouped_input = grouped_input.permute(0, 3, 1, 2) # (B, 3+C, M, K)
        # Chạy mini-PointNet (Conv2d hoạt động như Conv1d trên từng cụm)
        proposal_features = self.proposal_aggregator(grouped_input) # (B, 256, M, K)
        proposal_features = F.max_pool2d(proposal_features, kernel_size=(1, self.num_group_points)) # (B, 256, M, 1)
        proposal_features = proposal_features.squeeze(-1) # (B, 256, M)

        # 5. Dự đoán cuối cùng
        predictions = self.proposal_head(proposal_features) # (B, 1+3+3, M)
        
        # Tách các đầu ra
        pred_objectness = predictions[:, 0, :]   # (B, M)
        pred_center_res = predictions[:, 1:4, :] # (B, 3, M)
        pred_orient = predictions[:, 4:7, :]   # (B, 3, M)
        
        # Tính toán lại
        pred_objectness = pred_objectness.transpose(1, 0) # (M, B) - Lạ, nhưng VoteNet làm vậy
        pred_center = seed_xyz.transpose(2, 1) + pred_center_res # (B, 3, M)
        pred_orient = pred_orient # (B, 3, M)
        
        # Chuyển về định dạng (B, M, ...) cho dễ xử lý
        pred_objectness = pred_objectness.transpose(1, 0) # (B, M)
        pred_center = pred_center.permute(0, 2, 1)       # (B, M, 3)
        pred_orient = pred_orient.permute(0, 2, 1)       # (B, M, 3)
        
        # Chuẩn hóa vector pháp tuyến
        pred_orient_norm = F.normalize(pred_orient, p=2, dim=-1)

        return {
            'pred_votes': pred_votes,
            'pred_objectness': pred_objectness, # (B, M)
            'pred_center': pred_center,         # (B, M, 3)
            'pred_orientation': pred_orient_norm # (B, M, 3)
        }