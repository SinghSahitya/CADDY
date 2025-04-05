import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    # Ensure we don't try to sample more points than exist
    npoint = min(N, npoint)
    
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
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # Ensure we don't try to sample more points than exist
    nsample = min(N, nsample)
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # Handle edge cases where not enough points are found within radius
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points=None):
    """
    Input:
        npoint: number of points to sample
        radius: radius of ball
        nsample: number of samples in each local region
        xyz: input points position data, [B, N, 3] or [B, 3, N]
        points: input points data, [B, N, D] or [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    # Handle permuted dimensions (PointNet++ originally expects [B, N, C])
    if xyz.shape[1] == 3 and xyz.shape[2] != 3:
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
    
    if points is not None and points.shape[1] != xyz.shape[1]:
        points = points.permute(0, 2, 1)  # [B, N, D]
    
    B, N, C = xyz.shape
    
    # Ensure we don't try to sample more points than exist
    npoint = min(N, npoint)
    
    # Sample using farthest point sampling
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
    
    # Find nearest neighbors
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    
    # Group points
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)  # [B, npoint, nsample, 3]
    
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]
    
    return new_xyz, new_points

def sample_and_group_all(xyz, points=None):
    """
    Input:
        xyz: input points position data, [B, N, 3] or [B, 3, N]
        points: input points data, [B, N, D] or [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    # Handle permuted dimensions
    if xyz.shape[1] == 3 and xyz.shape[2] != 3:
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
    
    if points is not None and points.shape[1] != xyz.shape[1]:
        points = points.permute(0, 2, 1)  # [B, N, D]
    
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(xyz.device)  # [B, 1, 3]
    grouped_xyz = xyz.view(B, 1, N, C)  # [B, 1, N, 3]
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # [B, 1, N, 3+D]
    else:
        new_points = grouped_xyz  # [B, 1, N, 3]
    
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # Create convolutional layers for each MLP level
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        """
        Input:
            xyz: input points position data, [B, N, 3] or [B, 3, N]
            points: input points data, [B, N, D] or [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, npoint, 3]
            new_points: sample points feature data, [B, npoint, mlp[-1]]
        """
        # Record original dimension format
        xyz_format_transposed = xyz.shape[1] == 3 and xyz.shape[2] != 3
        
        # Standardize to [B, N, C] format for grouping operations
        if xyz_format_transposed:
            xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        
        if points is not None and points.shape[1] != xyz.shape[1]:
            points = points.permute(0, 2, 1)  # [B, N, D]
        
        # Sample and group points
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_points: [B, npoint, nsample, in_channel+3]
        # Transpose to [B, in_channel+3, nsample, npoint] for convolutions
        new_points = new_points.permute(0, 3, 2, 1)
        
        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Pooling across all points in the group to get a single feature vector per centroid
        new_points = torch.max(new_points, 2)[0]  # [B, mlp[-1], npoint]
        
        # Return to input format for consistency
        if xyz_format_transposed:
            return new_xyz.permute(0, 2, 1), new_points  # [B, 3, npoint], [B, mlp[-1], npoint]
        else:
            return new_xyz, new_points.permute(0, 2, 1)  # [B, npoint, 3], [B, npoint, mlp[-1]]

class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=10, normal_channel=False):
        super(PointNet2Classification, self).__init__()
        
        self.normal_channel = normal_channel
        in_channel = 6 if normal_channel else 3
        
        # First Set Abstraction layer
        self.sa1 = PointNetSetAbstraction(
            npoint=512, 
            radius=0.2, 
            nsample=32, 
            in_channel=in_channel, 
            mlp=[64, 64, 128], 
            group_all=False
        )
        
        # Second Set Abstraction layer
        self.sa2 = PointNetSetAbstraction(
            npoint=128, 
            radius=0.4, 
            nsample=32,  # Reduced from 64 to 32
            in_channel=128 + 3, 
            mlp=[128, 128, 256], 
            group_all=False
        )
        
        # Global Set Abstraction layer
        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True
        )
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Input:
            xyz: point cloud data, [B, 3, N] or [B, N, 3]
        Return:
            logits: classification logits, [B, num_classes]
        """
        # Ensure xyz is in [B, 3, N] format expected by our implementation
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)  # [B, 3, N]
        
        B, _, _ = xyz.shape
        
        # Extract normal channels if available
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        # Sequential Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        
        # l3_points has shape [B, 1024, 1]
        x = l3_points.view(B, 1024)
        
        # Fully connected layers with dropout
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=-1)
