import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# ============================================================
# PointNet++ Basic Modules (Không đổi)
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
# Point Transformer Layer (Không đổi)
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
# Set Abstraction Module (Không đổi)
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
        B, N, C = xyz.shape

        if self.npoint is None:
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            if points is not None:
                new_points = torch.cat([xyz, points], dim=2)
            else:
                new_points = xyz
            new_points = new_points.permute(0, 2, 1)
            new_points = new_points.unsqueeze(-1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0]
            new_points = new_points.permute(0, 2, 1)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            new_points = new_points.permute(0, 3, 2, 1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0]
            new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

# ============================================================
# Feature Propagation Module (Không đổi)
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
# PS6D Network (Không đổi)
# ============================================================

class PS6DNetwork(nn.Module):
    def __init__(self, num_points=1024, feature_dim=128):
        super().__init__()
        self.num_points = num_points
        self.sa1 = SetAbstraction(512, 0.05, 32, 3, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.1, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(None, None, None, 128 + 3, [128, 128, 256])
        self.pt1 = PointTransformerLayer(64, 64)
        self.pt2 = PointTransformerLayer(128, 128)
        self.fp3 = FeaturePropagation(384, [256, 128])
        self.fp2 = FeaturePropagation(192, [128, 64])
        self.fp1 = FeaturePropagation(67, [64, 64, feature_dim])
        self.centroid_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.rotation_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 4, 1)
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l1_points = self.pt1(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.pt2(l2_points.permute(0, 2, 1)).permute(0, 2, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, xyz, l1_points)
        feat = l0_points.permute(0, 2, 1)
        centroid_offset = self.centroid_head(feat).permute(0, 2, 1)
        quaternion = self.rotation_head(feat).permute(0, 2, 1)
        quaternion = F.normalize(quaternion, p=2, dim=-1)
        return centroid_offset, quaternion

# ============================================================
# Loss Functions Helper (Không đổi)
# ============================================================
def quaternion_to_matrix(quaternions):
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    xx = x * x; xy = x * y; xz = x * z; xw = x * w
    yy = y * y; yz = y * z; yw = y * w
    zz = z * z; zw = z * w
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
# *** SỬA ĐỔI PS6DLoss TẠI ĐÂY ***
# ============================================================

class PS6DLoss(nn.Module):
    def __init__(self, weight_translation=1.0, weight_rotation=1.0):
        super().__init__()
        self.weight_translation = weight_translation
        self.weight_rotation = weight_rotation

    def translation_loss(self, pred_offset, gt_offset, points, gt_centroid):
        """
        Center Distance Sensitive Loss (Phương trình 3)
        (Hàm này không đổi và vẫn được sử dụng)
        """
        dist_to_center = torch.norm(points - gt_centroid.unsqueeze(1), dim=-1)  # [B, N]
        max_dist = dist_to_center.max(dim=1, keepdim=True)[0]
        normalized_dist = dist_to_center / (max_dist + 1e-8)
        weight = 1.5 - normalized_dist  # [B, N]
        weight = torch.clamp(weight, 0.5, 1.5).unsqueeze(-1)  # [B, N, 1]
        loss = torch.abs(pred_offset - gt_offset) * weight
        return loss.mean()

    def rotation_loss_full_symmetry(self, pred_quat, gt_quat, model_points, sym_matrices, inf_sym_vec):
        """
        Symmetry-Aware Rotation Loss (Phương trình 2)
        (Hàm này vẫn được giữ lại, nhưng không được gọi trong forward() nữa
         vì chúng ta thiếu 3 tham số cuối)
        """
        B, N, _ = pred_quat.shape
        K, _ = model_points.shape
        Num_S, _, _ = sym_matrices.shape
        R_pred = quaternion_to_matrix(pred_quat)
        R_gt = quaternion_to_matrix(gt_quat)
        R_gt_expanded = R_gt.unsqueeze(2)
        sym_matrices_expanded = sym_matrices.view(1, 1, Num_S, 3, 3)
        R_gt_sym = torch.matmul(R_gt_expanded, sym_matrices_expanded)
        model_points_T = model_points.transpose(0, 1)
        P_pred = torch.matmul(R_pred, model_points_T)
        P_gt_sym_all = torch.matmul(R_gt_sym, model_points_T)
        P_pred_expanded = P_pred.unsqueeze(2)
        diffs = P_pred_expanded - P_gt_sym_all
        inf_sym_vec_T = inf_sym_vec.view(1, 1, 1, 3, 1)
        diffs_masked = diffs * inf_sym_vec_T
        dist_per_sym = torch.mean(torch.norm(diffs_masked, p=2, dim=3), dim=3)
        min_dist, _ = torch.min(dist_per_sym, dim=2)
        loss = torch.mean(min_dist)
        return loss


    # *** SỬA ĐỔI HÀM FORWARD ***
    def forward(self,
                # Đầu ra của mạng
                pred_offset, pred_quat,
                # Ground truth
                gt_offset, gt_quat,
                # Dữ liệu gốc
                points, gt_centroid
                # *** XÓA 3 THAM SỐ METADATA GÂY LỖI ***
                # model_points, sym_matrices, inf_sym_vec
                ):

        # 1. Tính loss translation (Không đổi)
        loss_t = self.translation_loss(pred_offset, gt_offset, points, gt_centroid)

        # 2. Tính loss rotation (Phiên bản đơn giản hóa)
        if self.weight_rotation > 0:
            # Do dataset (ps6d_dataset.py) không cung cấp metadata (model_points, v.v.)
            # chúng ta không thể gọi hàm `rotation_loss_full_symmetry`.
            # Thay vào đó, chúng ta dùng L1 loss đơn giản trên quaternion.
            
            # gt_quat từ dataset có shape [B, N, 4], giống pred_quat
            loss_r = F.l1_loss(pred_quat, gt_quat)

            # (Lưu ý: Một loss tốt hơn có thể xét cả q và -q,
            #  nhưng L1 đơn giản sẽ khắc phục được lỗi TypeError)
        else:
            loss_r = torch.tensor(0.0).to(pred_offset.device)

        # 3. Tổng loss (Phương trình 1)
        total_loss = self.weight_translation * loss_t + self.weight_rotation * loss_r

        # Trả về loss (Sửa loss_rotation để trả về .item() hoặc 0.0)
        loss_r_item = loss_r.item() if isinstance(loss_r, torch.Tensor) else 0.0
        
        return total_loss, {'loss_translation': loss_t.item(), 'loss_rotation': loss_r_item}