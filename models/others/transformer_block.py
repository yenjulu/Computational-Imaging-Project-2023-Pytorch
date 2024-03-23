# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
# from timm.models.layers import DropPath
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.to(device)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x.to(device)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# def get_sinusoid_encoding(n_position, d_hid):
#     ''' Sinusoid position encoding table '''

#     def get_position_angle_vec(position):
#         return [position / torch.pow(torch.tensor(10000.0), (2 * (hid_j // 2) / d_hid)).float() for hid_j in range(d_hid)]
    
#     sinusoid_table = torch.zeros(n_position, d_hid, device=device)
#     # sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#     for pos_i in range(n_position):
#         sinusoid_table[pos_i] = torch.tensor(get_position_angle_vec(pos_i))
        
#     sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
#     sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#     return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    
    def get_position_angle_vec(position):
        position = torch.tensor([position], dtype=torch.float, device=device)
        # Compute the divisors for each dimension
        div_term = torch.pow(torch.tensor(10000.0, device=device), torch.arange(0, d_hid, 2, dtype=torch.float, device=device) / d_hid)
        angle_vec = torch.zeros(d_hid, device=device)
        angle_vec[0::2] = position / div_term
        angle_vec[1::2] = position / div_term
        return angle_vec

    sinusoid_table = torch.zeros(n_position, d_hid, device=device)
    for pos_i in range(n_position):
        sinusoid_table[pos_i, :] = get_position_angle_vec(pos_i)
        
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # Apply sin to even indices
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # Apply cos to odd indices

    return sinusoid_table.unsqueeze(0)
