import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, 
                 query_dim, 
                 key_value_dim, 
                 heads=8, 
                 dim_head=64, 
                 dropout=0.0):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        # Linear layers for query, key, and value
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(key_value_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(key_value_dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),  # Match output to query channel dimension
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        # Reshape query from (1, 160, 256, 256) to (1, 65536, query_dim)
        batch_size, query_channels, height, width = query.shape
        spatial_size = height * width
        query_flat = query.view(batch_size, query_channels, spatial_size).permute(0, 2, 1)  # (1, 65536, query_dim)

        # Generate query, key, and value projections
        query_proj = self.to_q(query_flat)  # (1, 65536, inner_dim)
        key_proj = self.to_k(key_value)  # (1, 77, inner_dim)
        value_proj = self.to_v(key_value)  # (1, 77, inner_dim)

        # Reshape for multi-head attention
        query_proj = query_proj.view(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (1, heads, 65536, dim_head)
        key_proj = key_proj.view(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (1, heads, 77, dim_head)
        value_proj = value_proj.view(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (1, heads, 77, dim_head)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_proj, key_proj.transpose(-2, -1)) * (self.dim_head ** -0.5)  # (1, heads, 65536, 77)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attention = torch.matmul(attn_weights, value_proj)  # (1, heads, 65536, dim_head)

        # Combine heads
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, -1, self.inner_dim)  # (1, 65536, inner_dim)

        # Final projection
        attention = self.to_out(attention)  # (1, 65536, query_channels)

        # Reshape output back to (1, 160, 256, 256)
        attention = attention.permute(0, 2, 1).view(batch_size, query_channels, height, width)
        return attention

# Example usage
query = torch.randn(1, 320, 128, 128)  # Query tensor with shape (batch, channels, height, width)
key_value = torch.randn(1, 77, 1024)  # Key and value tensor with shape (batch, seq_len, key_value_dim)

cross_attention = CrossAttention(query_dim=query.shape[1], key_value_dim=1024, heads=8, dim_head=64)
output = cross_attention(query, key_value)
print(output.shape)  # Should be (1, 160, 256, 256)