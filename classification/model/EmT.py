# This is the network script
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 将此代码段添加到 EmT.py 的 import 部分之后

# SEED (62通道) 对称索引对 (Left_Index, Right_Index)
SEED_PAIRS = [
    (0, 2), (3, 4), (5, 13), (6, 12), (7, 11), (8, 10), (15, 21), (16, 20), 
    (17, 19), (23, 31), (24, 30), (25, 29), (26, 28), (32, 40), (33, 39), 
    (34, 38), (35, 37), (41, 49), (42, 48), (43, 47), (44, 46), (50, 57), 
    (51, 56), (52, 55), (58, 61), (59, 60) 
    # 中线通道如 FPZ, FZ, CZ... 被忽略
]

# FACED/THU-EP (32通道) 对称索引对
FACED_PAIRS = [
    (0, 1),   # Fp1-Fp2
    (3, 4),   # F3-F4
    (5, 6),   # F7-F8
    (7, 8),   # FC1-FC2
    (9, 10),  # FC5-FC6
    (12, 13), # C3-C4
    (14, 15), # T7-T8
    (18, 19), # CP1-CP2
    (20, 21), # CP5-CP6
    (23, 24), # P3-P4
    (25, 26), # P7-P8
    (27, 28), # PO3-PO4
    (30, 31)  # O1-O2
]

class GraphConvolution(Module):
    """
    LGG-specific GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_fliter(K, in_feature, out_feature)

    def init_fliter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias == True:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        # we approximate lambda_max ~= 2
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    def rescale_L(self, L):
        largest_eigval, _ = torch.linalg.eigh(L)
        largest_eigval = torch.max(largest_eigval)
        L = (2. / largest_eigval) * L - torch.eye(L.size(0), device=L.device, dtype=torch.float)
        return L

    def chebyshev(self, x, L):
        # to do graph convolution here] X_0 = X, X_1 = L.X, X_k = 2.L.X_(k-1) - X_(k-2)
        x1 = torch.matmul(L, x)
        x_ = torch.stack((x, x1), dim=1)  # (b, 2, chan, fin)
        if self.K > 1:
            for k in range(2, self.K):
                x_current = 2 * torch.matmul(L, x_[:, -1]) - x_[:, -2]  # X_k = 2.L.X_(k-1) - X_(k-2)
                x_current = x_current.unsqueeze(dim=1)
                x_ = torch.cat((x_, x_current), dim=1)

        x_ = x_.permute(1, 0, 2, 3)  # (k, b, chan, fin)   w: (k, 1, fin, fout) f:
        out = torch.matmul(x_, self.filter_weight)  # (k, b, chan, fout)
        out = torch.sum(out, dim=0)  # (b, chan, fout)
        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        # x: (b, chan, f) adj
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        out = (out, adj)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K,
                 graph2token='Linear', encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.graph2token = graph2token
        self.K = K  # useful for ChebyNet
        assert graph2token in ['Linear', 'AvgPool', 'MaxPool', 'Flatten'], "graph2vector type is not supported!"
        if graph2token == 'Linear':
            self.tokenizer = nn.Linear(num_node*out_features, out_features)
        else:
            self.tokenizer = None
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ['Cheby', 'GCN'], "encoder type is not supported!"
        if encoder_type == 'GCN':
            GNN = GCN(in_features, out_features)
        if encoder_type == 'Cheby':
            GNN = ChebyNet(self.K, in_features, out_features)
        return GNN

    def forward(self, x, adj):
        # x: b, channel, feature
        # adj: m, n, n
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            if self.graph2token == 'AvgPool':
                output = torch.mean(x, dim=-1)
            elif self.graph2token == 'MaxPool':
                output = torch.max(x, dim=-1)[0]
            else:
                output = x.view(x.size(0), -1)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, anchor=3, dropout = 0., alpha=0.25):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.STA = nn.Sequential(
            nn.Dropout(alpha * dropout),
            weight_norm(nn.Conv2d(self.heads, self.heads, (anchor, 1),
                                  stride=1, padding=self.get_padding(anchor))),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = self.STA(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def get_padding(self, kernel):
        return (int(0.5 * (kernel - 1)), 0)


class TTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., alpha=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, alpha=alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViewAttention(nn.Module):
    def __init__(self, input_dim):
        super(ViewAttention, self).__init__()
        # 一个简单的 MLP 来计算每个视图的重要性分数
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, num_views, hidden_dim)
        
        # 1. 计算每个视图的分数
        scores = self.score_net(x)  # (batch, num_views, 1)
        
        # 2. 归一化权重 (Softmax over views)
        weights = F.softmax(scores, dim=1)
        
        # 3. 加权求和
        # (batch, num_views, 1) * (batch, num_views, hidden_dim) -> sum -> (batch, hidden_dim)
        fused = torch.sum(weights * x, dim=1)
        
        return fused, weights # 返回 weights 用于可视化（可选）

class AsymmetryEncoder(nn.Module):
    def __init__(self, pairs, in_features, out_features):
        super(AsymmetryEncoder, self).__init__()
        self.pairs = pairs
        self.num_pairs = len(pairs)
        
        # 用于将差分特征投影到 hidden_graph 维度
        # 输入维度 = 差分通道数 * 特征数
        self.projector = nn.Sequential(
            nn.Linear(self.num_pairs * in_features, out_features * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(out_features * 2, out_features)
        )

    def forward(self, x):
        # x shape: (batch, num_channels, num_features)
        
        # 1. 提取左脑和右脑特征
        left_indices = [p[0] for p in self.pairs]
        right_indices = [p[1] for p in self.pairs]
        
        x_left = x[:, left_indices, :]
        x_right = x[:, right_indices, :]
        
        # 2. 计算差分特征 (Left - Right)
        # 这代表了半球间的不对称性 (Hemispheric Asymmetry)
        x_diff = x_left - x_right # (batch, num_pairs, num_features)
        
        # 3. 展平并投影
        x_diff = x_diff.contiguous().view(x_diff.size(0), -1)
        out = self.projector(x_diff) # (batch, hidden_graph)
        
        return out

class EmT(nn.Module):
    def __init__(self, layers_graph=[1, 2], layers_transformer=1, num_adj=3, num_chan=62,
                 num_feature=5, hidden_graph=16, K=2, num_head=8, dim_head=16,
                 dropout=0.25, num_class=3, alpha=0.25, graph2token='Linear', encoder_type='GCN'):
        super(EmT, self).__init__()
        self.graph_encoder_type = encoder_type
        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K, graph2token=graph2token, encoder_type=encoder_type
        )

        self.adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)
# ================= [改进 1：不对称分支初始化] =================
        # 根据通道数自动选择 Pair 列表
        if num_chan == 62:
            pairs = SEED_PAIRS
        elif num_chan == 32:
            pairs = FACED_PAIRS
        else:
            # 如果是其他通道数，默认回退到不使用不对称分支，或者报错
            print(f"Warning: No symmetry pairs defined for {num_chan} channels. Asymmetry branch disabled.")
            pairs = []
        self.use_asymmetry = len(pairs) > 0
        if self.use_asymmetry:
            self.asym_encoder = AsymmetryEncoder(pairs, num_feature, hidden_graph)
        
        if graph2token in ['AvgPool', 'MaxPool']:
            hidden_graph = num_chan
        if graph2token == 'Flatten':
            hidden_graph = num_chan*hidden_graph

        self.view_attention = ViewAttention(hidden_graph)

        self.transformer = TTransformer(
            depth=layers_transformer,
            dim=hidden_graph, heads=num_head,
            dim_head=dim_head, dropout=dropout, mlp_dim=dim_head,
            alpha=alpha
        )

        self.to_GNN_out = nn.Linear(num_chan*num_feature, hidden_graph, bias=False)

        self.MLP = nn.Sequential(
            nn.Linear(hidden_graph, num_class)
        )

    def forward(self, x):
        # x: batch, sequence, chan, feature
        b, s, chan, f = x.size()
        x = rearrange(x, 'b s c f  -> (b s) c f')
        if self.graph_encoder_type == 'Cheby':
            adjs = self.get_adj(self_loop=False)
        else:
            adjs = self.get_adj()

# 1. 计算原始的多视图 GCN 特征
        x_flat = x.contiguous().view(x.size(0), -1) # 改名 x_ -> x_flat 避免混淆
        token_base = self.to_GNN_out(x_flat)        # 基础视图 (Linear projection)
        token_g1 = self.GE1(x, adjs[0])             # GCN视图 1
        token_g2 = self.GE2(x, adjs[1])             # GCN视图 2
        
        # 2. 构建视图列表
        views = [token_base, token_g1, token_g2]
        
        # 3. [改进 1] 计算不对称特征视图
        if self.use_asymmetry:
            token_asym = self.asym_encoder(x)
            views.append(token_asym)
            
        # 4. [改进 2] 动态注意力融合
        # 堆叠所有视图: (batch*seq, num_views, hidden_graph)
        x_stack = torch.stack(views, dim=1) 
        
        # 使用注意力代替 mean
        # 原代码: x = torch.mean(x, dim=1)
        x, attn_weights = self.view_attention(x_stack) # x shape: (batch*seq, hidden_graph)

        # temporal contextual transformer
        x = rearrange(x, '(b s) h -> b s h', b=b, s=s)
        x = self.transformer(x)
        x = torch.mean(x, dim=-2)
        x = self.MLP(x)
        return x

    def get_adj(self, self_loop=True):
        # self.adjs : n, node, node
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        return adj


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 8, 62, 7))
    emt = EmT(layers_graph=[1, 2], layers_transformer=4, num_adj=2,
              num_chan=62, num_feature=7, hidden_graph=32,
              K=4, num_head=16, dim_head=32, dropout=0.25, num_class=2,
              graph2token='Linear', encoder_type='Cheby', alpha=0.25)
    print(emt)
    print(count_parameters(emt))

    out = emt(data)
    print('Done')




