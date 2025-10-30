import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# ============================================================
# PointNet++ Basic Modules
# ============================================================

def square_distance(src, dst):
    """Calculate Euclidean distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """Farthest Point Sampling"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Query ball point"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """Sample and group points"""
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, C))

    idx = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = torch.gather(xyz, 1, idx.reshape(B, -1, 1).expand(-1, -1, C)).reshape(B, S, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = torch.gather(points, 1, idx.reshape(B, -1, 1).expand(-1, -1, points.size(-1))).reshape(B, S, nsample, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points

# ============================================================
# Point Transformer Layer
# ============================================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_q = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_k = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_v = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, N]
        q = self.conv_q(x)  # [B, C', N]
        k = self.conv_k(x)
        v = self.conv_v(x)

        # Self-attention
        attn = torch.softmax(torch.matmul(q.transpose(1, 2), k) / np.sqrt(q.size(1)), dim=-1)
        out = torch.matmul(v, attn.transpose(1, 2))

        out = self.conv_out(out)
        out = self.bn(out)
        return F.relu(out)

# ============================================================
# Set Abstraction Module
# ============================================================

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        # *** BẮT ĐẦU SỬA LỖI ***
        B, N, C = xyz.shape

        if self.npoint is None:
            # --- Logic cho Global Set Abstraction (npoint=None) ---
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device) # Tạo centroid giả

            # Ghép xyz và features
            if points is not None:
                new_points = torch.cat([xyz, points], dim=2) # [B, N, 3 + C_in]
            else:
                new_points = xyz # [B, N, 3]

            # Thay đổi kích thước cho MLPs (Conv2d)
            new_points = new_points.permute(0, 2, 1) # [B, 3 + C_in, N]
            new_points = new_points.unsqueeze(-1) # [B, 3 + C_in, N, 1] (coi như nsample=1)

            # Áp dụng MLPs
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points))) # [B, C_out, N, 1]

            # Global Max Pooling (lấy max qua N điểm)
            new_points = torch.max(new_points, 2)[0] # [B, C_out, 1]
            new_points = new_points.permute(0, 2, 1) # [B, 1, C_out]

        else:
            # --- Logic cũ cho Local Set Abstraction (npoint=integer) ---
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            new_points = new_points.permute(0, 3, 2, 1)  # [B, C+3, nsample, npoint]

            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points))) # [B, C_out, nsample, npoint]

            # Max Pooling (lấy max qua nsample)
            new_points = torch.max(new_points, 2)[0] # [B, C_out, npoint]
            new_points = new_points.permute(0, 2, 1) # [B, npoint, C_out]

        # *** KẾT THÚC SỬA LỖI ***

        return new_xyz, new_points

# ============================================================
# Feature Propagation Module
# ============================================================

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N, 3] - points with more samples
        xyz2: [B, M, 3] - points with fewer samples
        points1: [B, N, C1]
        points2: [B, M, C2]
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape

        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                torch.gather(points2, 1, idx.reshape(B, -1, 1).expand(-1, -1, points2.size(-1))).reshape(B, N, 3, -1) * weight.view(B, N, 3, 1),
                dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)

# ============================================================
# PS6D Network
# ============================================================

class PS6DNetwork(nn.Module):
    def __init__(self, num_points=1024, feature_dim=128):
        super().__init__()
        self.num_points = num_points

        # PointNet++ Backbone
        self.sa1 = SetAbstraction(512, 0.05, 32, 3, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.1, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(None, None, None, 128 + 3, [128, 128, 256])  # Global

        # Point Transformer layers
        self.pt1 = PointTransformerLayer(64, 64)
        self.pt2 = PointTransformerLayer(128, 128)

        # Feature Propagation
        self.fp3 = FeaturePropagation(384, [256, 128])
        self.fp2 = FeaturePropagation(192, [128, 64])
        self.fp1 = FeaturePropagation(67, [64, 64, feature_dim])

        # Centroid Regression Head
        self.centroid_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # (x, y, z) offset to centroid
        )

        # Rotation Prediction Head (Quaternion)
        self.rotation_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 4, 1)  # Quaternion (w, x, y, z)
        )

    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3] - normalized point cloud
        Returns:
            centroid_offset: [B, N, 3] - offset from each point to centroid
            quaternion: [B, N, 4] - rotation quaternion for each point
        """
        B, N, _ = xyz.shape

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l1_points = self.pt1(l1_points.permute(0, 2, 1)).permute(0, 2, 1)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.pt2(l2_points.permute(0, 2, 1)).permute(0, 2, 1)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, xyz, l1_points)

        # Per-point predictions
        feat = l0_points.permute(0, 2, 1)  # [B, C, N]

        centroid_offset = self.centroid_head(feat).permute(0, 2, 1)  # [B, N, 3]
        quaternion = self.rotation_head(feat).permute(0, 2, 1)  # [B, N, 4]

        # Normalize quaternion
        quaternion = F.normalize(quaternion, p=2, dim=-1)

        return centroid_offset, quaternion

# ============================================================
# Loss Functions
# ============================================================
def quaternion_to_matrix(quaternions):
    """
    Hàm hỗ trợ chuyển đổi một batch quaternion sang ma trận xoay.
    Đầu vào: [..., 4] (w, x, y, z)
    Đầu ra: [..., 3, 3]
    """
    # Đảm bảo chuẩn hóa
    quaternions = F.normalize(quaternions, p=2, dim=-1)

    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    # Tính toán các thành phần của ma trận
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w

    yy = y * y
    yz = y * z
    yw = y * w

    zz = z * z
    zw = z * w

    # Tạo ma trận
    matrix = torch.empty(quaternions.shape[:-1] + (3, 3), device=quaternions.device)

    matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 0, 2] = 2 * (xz + yw)

    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    matrix[..., 1, 2] = 2 * (yz - xw)

    matrix[..., 2, 0] = 2 * (xz - yw)
    matrix[..., 2, 1] = 2 * (yz + xw)
    matrix[..., 2, 2] = 1 - 2 * (xx + yy)

    return matrix

# ============================================================
# Loss Functions (Phiên bản đầy đủ theo Paper)
# ============================================================

class PS6DLoss(nn.Module):
    def __init__(self, weight_translation=1.0, weight_rotation=1.0):
        super().__init__()
        self.weight_translation = weight_translation
        self.weight_rotation = weight_rotation

    def translation_loss(self, pred_offset, gt_offset, points, gt_centroid):
        """
        Center Distance Sensitive Loss (Phương trình 3)
        [cite: 31-34, 181, 183-184]
        """
        # Tính toán trọng số dựa trên khoảng cách (gần tâm hơn = trọng số cao hơn)
        dist_to_center = torch.norm(points - gt_centroid.unsqueeze(1), dim=-1)  # [B, N]
        max_dist = dist_to_center.max(dim=1, keepdim=True)[0]

        # Chuẩn hóa khoảng cách
        normalized_dist = dist_to_center / (max_dist + 1e-8)

        # Trọng số: điểm gần tâm có trọng số gần 1.5, điểm xa tâm có trọng số gần 0.5
        weight = 1.5 - normalized_dist  # [B, N]
        weight = torch.clamp(weight, 0.5, 1.5).unsqueeze(-1)  # [B, N, 1]

        # Weighted L1 loss
        loss = torch.abs(pred_offset - gt_offset) * weight
        return loss.mean()

    def rotation_loss(self, pred_quat, gt_quat, model_points, sym_matrices, inf_sym_vec):
        """
        Symmetry-Aware Rotation Loss (Phương trình 2)
        [cite: 164-168, 172-178, 180-182]

        Args:
            pred_quat: [B, N, 4] - Quaternion dự đoán (w,x,y,z)
            gt_quat: [B, N, 4] - Quaternion ground-truth (w,x,y,z)
            model_points: [K, 3] - Đám mây điểm của model CAD (K điểm)
            sym_matrices: [Num_S, 3, 3] - Danh sách các ma trận xoay đối xứng S
            inf_sym_vec: [3] - Vector chỉ định đối xứng vô hạn (ví dụ: [1,1,1] cho không đối xứng,
                             [0,0,1] cho đối xứng quanh trục Z, [0,0,0] cho hình cầu)
        """

        B, N, _ = pred_quat.shape
        K, _ = model_points.shape
        Num_S, _, _ = sym_matrices.shape

        # 1. Chuyển đổi quaternion sang ma trận xoay [cite: 172-173]
        R_pred = quaternion_to_matrix(pred_quat)  # [B, N, 3, 3]
        R_gt = quaternion_to_matrix(gt_quat)      # [B, N, 3, 3]

        # 2. Chuẩn bị cho phép nhân ma trận (broadcasting)
        # Mở rộng R_gt và sym_matrices để tính toán tất cả các tư thế đối xứng GT
        R_gt_expanded = R_gt.unsqueeze(2)                # [B, N, 1, 3, 3]
        sym_matrices_expanded = sym_matrices.view(1, 1, Num_S, 3, 3) # [1, 1, Num_S, 3, 3]

        # R_gt_sym: [B, N, Num_S, 3, 3] - Tất cả các tư thế ground-truth hợp lệ [cite: 174-175]
        R_gt_sym = torch.matmul(R_gt_expanded, sym_matrices_expanded)

        # 3. Áp dụng các phép xoay lên các điểm model
        model_points_T = model_points.transpose(0, 1) # [3, K]

        # P_pred: [B, N, 3, K] - Các điểm model được xoay bởi R_pred
        P_pred = torch.matmul(R_pred, model_points_T)

        # P_gt_sym_all: [B, N, Num_S, 3, K] - Các điểm model được xoay bởi tất cả các R_gt_sym
        P_gt_sym_all = torch.matmul(R_gt_sym, model_points_T)

        # 4. Tính toán khoảng cách
        # Mở rộng P_pred để so sánh với tất cả các P_gt_sym
        P_pred_expanded = P_pred.unsqueeze(2) # [B, N, 1, 3, K]

        # diffs: [B, N, Num_S, 3, K] - Sai lệch giữa dự đoán và tất cả các GT đối xứng
        diffs = P_pred_expanded - P_gt_sym_all

        # 5. Áp dụng vector đối xứng vô hạn 'v' [cite: 177-178]
        # Chúng ta mask các chiều không quan trọng (ví dụ: x, y cho đối xứng trục z)
        inf_sym_vec_T = inf_sym_vec.view(1, 1, 1, 3, 1) # [1, 1, 1, 3, 1]
        diffs_masked = diffs * inf_sym_vec_T

        # 6. Tính L2 norm (khoảng cách) trên các điểm (dim K) và tọa độ (dim 3)
        # dist_per_sym: [B, N, Num_S] - Khoảng cách L2 trung bình cho mỗi tư thế đối xứng
        # Chúng ta dùng mean() thay vì sum() để chuẩn hóa theo số lượng điểm K
        dist_per_sym = torch.mean(torch.norm(diffs_masked, p=2, dim=3), dim=3)

        # 7. Tìm khoảng cách nhỏ nhất (min) trong số các đối xứng [cite: 175]
        # min_dist: [B, N]
        min_dist, _ = torch.min(dist_per_sym, dim=2)

        # 8. Lấy trung bình trên toàn bộ batch và các điểm
        loss = torch.mean(min_dist)
        return loss


    def forward(self,
                # Đầu ra của mạng
                pred_offset, pred_quat,
                # Ground truth
                gt_offset, gt_quat,
                # Dữ liệu gốc
                points, gt_centroid,
                # Siêu dữ liệu (Metadata) cho loss xoay
                model_points, sym_matrices, inf_sym_vec
                ):

        # Tính loss translation (Phương trình 3) [cite: 36, 150-151]
        loss_t = self.translation_loss(pred_offset, gt_offset, points, gt_centroid)

        # Tính loss rotation (Phương trình 2) [cite: 36, 150-151]
        if self.weight_rotation > 0:
            loss_r = self.rotation_loss(pred_quat, gt_quat, model_points, sym_matrices, inf_sym_vec)
        else:
            loss_r = torch.tensor(0.0).to(pred_offset.device)

        # Tổng loss (Phương trình 1) [cite: 36, 150-151]
        total_loss = self.weight_translation * loss_t + self.weight_rotation * loss_r

        return total_loss, {'loss_translation': loss_t.item(), 'loss_rotation': loss_r.item()}