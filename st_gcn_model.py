# st_gcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """Khối xử lý KHÔNG GIAN (Spatial) - Giữ nguyên"""
    def __init__(self, in_channels, out_channels, A, bias=True):
        super(GraphConv, self).__init__()
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        if self.A.device != x.device:
            self.A = self.A.to(x.device)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        return self.conv(x)

class TemporalConv(nn.Module):
    """
    Khối xử lý THỜI GIAN (Temporal) - MỚI
    Dùng Conv2d với kernel (9, 1) nghĩa là quét qua 9 frame liên tiếp trên 1 khớp.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class STGCN_Block(nn.Module):
    """Kết hợp Spatial + Temporal + Residual"""
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(STGCN_Block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels)
        
        # Residual connection (nếu in != out thì cần conv 1x1 để khớp kích thước)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = lambda x: x
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class STGCN(nn.Module):
    """Kiến trúc tổng thể nâng cấp"""
    def __init__(self, in_channels, num_class, A):
        super(STGCN, self).__init__()
        
        # Batch Norm cho dữ liệu đầu vào (giúp model hội tụ nhanh hơn)
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])

        self.layers = nn.ModuleList([
            STGCN_Block(in_channels, 64, A),
            STGCN_Block(64, 64, A),
            STGCN_Block(64, 128, A),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 256, A)
        ])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x input ban đầu: (N, C, T, V, M)
        
        # Bước 1: Xóa chiều M (người) vì chỉ có 1 người -> x còn (N, C, T, V)
        x = x.squeeze(-1) 
        
        # Lấy kích thước hiện tại
        N, C, T, V = x.size() 
        
        # --- SỬA LỖI TẠI ĐÂY ---
        # Input đang là 4 chiều: (N, C, T, V) tương ứng index (0, 1, 2, 3)
        # Muốn đổi thành (N, V, C, T) thì phải gọi index (0, 3, 1, 2)
        # Sau đó .view(N, V * C, T) để gộp V và C lại cho BatchNorm
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        
        # Chạy qua Batch Normalization
        x = self.data_bn(x)
        
        # Trả lại shape cũ (N, C, T, V) để đưa vào các layer GCN
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # Chạy qua các layer ST-GCN
        for layer in self.layers:
            x = layer(x)
            
        # Pooling và Phân loại
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x