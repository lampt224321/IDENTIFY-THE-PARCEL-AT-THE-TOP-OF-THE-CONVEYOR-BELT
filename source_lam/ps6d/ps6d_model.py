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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
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

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_q = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_k = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_v = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        q = self.conv_q(x); k = self.conv_k(x); v = self.conv_v(x)
        attn = torch.softmax(torch.matmul(q.transpose(1, 2), k) / np.sqrt(q.size(1)), dim=-1)
        out = torch.matmul(v, attn.transpose(1, 2)); out = self.conv_out(out); out = self.bn(out)
        return F.relu(out)

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint; self.radius = radius; self.nsample = nsample
        self.mlp_convs = nn.ModuleList(); self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, xyz, points):
        B, N, C = xyz.shape
        if self.npoint is None:
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.permute(0, 2, 1).unsqueeze(-1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]; new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            new_points = new_points.permute(0, 3, 2, 1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]; new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        return new_xyz, new_points

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList(); self.mlp_bns = nn.ModuleList()
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
            dists, idx = dists.sort(dim=-1); dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8); norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                torch.gather(points2, 1, idx.reshape(B, -1, 1).expand(-1, -1, points2.size(-1))).reshape(B, N, 3, -1) * weight.view(B, N, 3, 1),
                dim=2)
        new_points = torch.cat([points1, interpolated_points], dim=-1) if points1 is not None else interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]; new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

# ============================================================
# PS6D Network
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
            nn.Conv1d(feature_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

        # Đầu ra là 3 (cho vector pháp tuyến) thay vì 4 (cho quaternion)
        self.rotation_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 3, 1) # <<< Thay đổi từ 4 thành 3
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

        #  Lấy vector pháp tuyến và chuẩn hóa nó 
        pred_normal = self.rotation_head(feat).permute(0, 2, 1)
        pred_normal = F.normalize(pred_normal, p=2, dim=-1) # Đảm bảo nó là vector đơn vị

        return centroid_offset, pred_normal

# ============================================================
# Loss Functions 
# ============================================================

class PS6DLoss(nn.Module):
    def __init__(self, weight_translation=1.0, weight_rotation=1.0):
        super().__init__()
        self.weight_translation = weight_translation
        self.weight_rotation = weight_rotation

    def translation_loss(self, pred_offset, gt_offset, points, gt_centroid):
        """
        Center Distance Sensitive Loss (Không đổi)
        """
        dist_to_center = torch.norm(points - gt_centroid.unsqueeze(1), dim=-1)
        max_dist = dist_to_center.max(dim=1, keepdim=True)[0]
        normalized_dist = dist_to_center / (max_dist + 1e-8)
        weight = 1.5 - normalized_dist
        weight = torch.clamp(weight, 0.5, 1.5).unsqueeze(-1)
        loss = torch.abs(pred_offset - gt_offset) * weight
        return loss.mean()

    # *** Orientation Loss (dựa trên Cosine Similarity) ***
    def orientation_loss(self, pred_n, gt_n):
        """
        Tính toán loss dựa trên công thức OE (arccos(n . n_hat)).
        Chúng ta sẽ tối ưu (1 - dot_product) thay vì acos để ổn định hơn.
        """
        # pred_n và gt_n đều đã được chuẩn hóa (normalize)
        # Tích vô hướng (dot product)
        dot_product = torch.sum(pred_n * gt_n, dim=-1) # Shape: [B, N]

        # Kẹp giá trị để tránh lỗi số học
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Cosine Similarity Loss: (1 - cos(theta))
        # Tối ưu loss này sẽ làm cho cos(theta) -> 1, tức là theta -> 0
        loss = 1.0 - dot_product
        return loss.mean()

    def forward(self,
                pred_offset, pred_normal, 
                gt_offset, gt_normal,     
                points, gt_centroid
                ):

        loss_t = self.translation_loss(pred_offset, gt_offset, points, gt_centroid)

        if self.weight_rotation > 0:
            loss_r = self.orientation_loss(pred_normal, gt_normal)
        else:
            loss_r = torch.tensor(0.0).to(pred_offset.device)

        # 3. Tổng loss
        total_loss = self.weight_translation * loss_t + self.weight_rotation * loss_r

        loss_r_item = loss_r.item() if isinstance(loss_r, torch.Tensor) else 0.0

        return total_loss, {'loss_translation': loss_t.item(), 'loss_rotation': loss_r_item}