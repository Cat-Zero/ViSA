""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from kiwisolver import strength

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from functools import reduce
from operator import mul
import copy

logger = logging.getLogger(__name__)
def softmax_one(x, dim=-1):
    return (x.exp() + 1e-6) / (x.exp().sum(dim, keepdim=True) + 1)

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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        downsample_rate: int = 1,   # 下采样
        smax_mode=None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # qkv获取
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.smax = smax_mode
    def _separate_heads(self, x, num_heads: int) :
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v) :
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # B,N_heads,N_tokens,C_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B,N_heads,N_tokens,C_per_head
        # Scale
        attn = attn / math.sqrt(c_per_head)
        if self.smax is not None:
            attn = softmax_one(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        # # B,N_tokens,C
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class OutAttenBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4.0,
                 activation = nn.ReLU,
                 attention_downsample_rate: int = 1,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        # self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward(self, q):
        # attn_out = self.cross_attn(q, k, v)
        # queries = q + attn_out
        # queries = self.norm1(queries)
        queries = q + self.attn(q)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


class PromptRecapBlock(nn.Module):
    def __init__(self,
                embedding_dim: int,
                num_heads: int,
                activation = nn.ReLU,
                mlp_ratio = 2.0,
                method='attn'):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = Attention(embedding_dim, num_heads)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.method = method

        # self.attn2 = Attention(embedding_dim, num_heads)

    def forward(self, queries, keys):
        # if self.method == 'cat':
        #     queries = torch.cat((queries, keys), dim=1)
        #     attn_out = self.attn2(queries)
        #     queries = queries + attn_out
        #     queries = self.norm1(queries)
        # elif self.method == 'add':
        #     queries = queries + keys
        #     queries = self.norm1(queries)

        attn_out = self.cross_attn(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm1(queries)
        attn_out = self.attn(queries)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        # if self.method == 'cat':
        #     k_len = keys.shape[1]
        #     # print(queries[:-k_len].shape)
        #     return queries[:, :-k_len]
        return queries

class Router(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        x_flat = x.squeeze(1)  # [B, D]

        logits = self.gate(x_flat)  # [B, num_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        probs = F.softmax(logits, dim=-1)  # [B, num_experts]
        importance = probs.mean(dim=0)
        importance = importance / (importance.sum() + 1e-6)
        load_balance_loss = self.num_experts * torch.sum(importance ** 2)

        return top_k_weights, top_k_indices, load_balance_loss

class MoEBlock(nn.Module):
    def __init__(self, num_heads, in_planes, prompt_len):
        super().__init__()
        # PRM init
        self.prompt = nn.Parameter(torch.zeros(1, prompt_len, in_planes))
        trunc_normal_(self.prompt, std=.02)
        self.prm = PromptRecapBlock(embedding_dim=in_planes, num_heads=num_heads)

    def forward(self, inv_features):
        # PRM
        query_feat = torch.repeat_interleave(self.prompt, inv_features.shape[0], dim=0)
        Re_Prompt = self.prm(query_feat, inv_features)
        return Re_Prompt


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class VisionTransformer_multiview(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0, inner_sub=True, local_feat=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token_sky = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token_ground = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        self.local_feat = local_feat
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.inner_sub = inner_sub

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_token'}

    def forward(self, x, camera_id, view_id):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        sky_mask = (view_id == 0)
        ground_mask = (view_id == 1)
        view_tokens = torch.zeros(B, 1, self.embed_dim, device=self.view_token_sky.device, dtype=self.view_token_sky.dtype)
        view_tokens[sky_mask] = self.view_token_sky
        view_tokens[ground_mask] = self.view_token_ground
        x = torch.cat((cls_tokens, view_tokens, x), dim=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        if self.local_feat:
            for blk in self.blocks[:-1]:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]
            return x

        else:
            for blk in self.blocks:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]

            x = self.norm(x)
            return x[:, 0].reshape(x.shape[0], -1, 1, 1), x[:, 1].reshape(x.shape[0], -1, 1, 1)

    def load_param(self, pretrain_path):
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = self.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed(v, self.pos_embed.data, self.patch_embed.num_y, self.patch_embed.num_x, 2)
                state_dict[k] = v
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

def resize_pos_embed(posemb, posemb_new, hight, width, cls_token_num):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :cls_token_num], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def graph_norm(A):
    # A = A + I    A: (bs, n, num_nodes, num_nodes)  [B, N_, P+1, P+1]
    # Degree
    d = A.sum(-1)  # (bs, n, num_nodes)
    # D = D^-1/2
    d = torch.pow(d, -0.5)
    D = A.detach().clone()
    for batch_index, batch in enumerate(A):
        for class_index, _ in enumerate(batch):
            D[batch_index, class_index] = torch.diag(d[batch_index, class_index])
    norm_A = torch.stack([D_i.bmm(A_i).bmm(D_i) for D_i, A_i in zip(D, A)]).detach()

    return norm_A


def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))


def cal_edge_emb(x, p=2, dim=-1):  # v1_graph---taking the similairty by
    '''
    x: [B N P+1 D]
    return: (B, N, P+1, P+1)
    '''
    # x = torch.stack([F.normalize(xi, p=p, dim=dim) for xi in x]).detach()  # [m+1, 1000, 1024], [100, 1024, 101]
    x = F.normalize(x, p=p, dim=dim).detach()
    x_r = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B N P+1 D]
    x = x.transpose(-1, -2)
    x_c = x.reshape(-1, x.shape[-2], x.shape[-1])

    A = torch.bmm(x_r, x_c).reshape(x.shape[0], x.shape[1], x.shape[-1], x.shape[-1]).detach()
    # A = torch.stack([torch.bmm(x_r_i, x_c_i) for (x_r_i, x_c_i) in zip(x_r, x_c)]).detach()

    return A


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, class_num=None, sparse_inputs=False, bias=False,
                 dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        # self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat  # [B, P, N+1, D]
        node_size = adj.size()[-1]
        adj = torch.clip(adj, min=0.0)  # [B, P, N+1, N+1]
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0)
        adj = adj + I  # [B, P, N+1, N+1]
        adj = graph_norm(adj)
        pre_sup = torch.matmul(x, self.gcn_weights)
        output = torch.matmul(adj, pre_sup)
        # output += self.gcn_bias.unsqueeze(1)

        return self.act(output[:, :, 0, :])


class Vision_Transformer_ViSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        input_size = cfg.INPUT.SIZE_TRAIN
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        depth = cfg.MODEL.BACKBONE.DEPTH
        sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
        stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
        drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
        drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
        attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
        inner_sub = cfg.MODEL.BACKBONE.INNER_SUB
        self.in_planes = 768
        self.prompt_len = cfg.MODEL.BACKBONE.PROMPT_LEN
        self.prompt_trans_depth = cfg.MODEL.BACKBONE.PROMPT_DEPTH
        # self.use_prompt = cfg.MODEL.BACKBONE.USE_PROMPT
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        self.use_prm = cfg.MODEL.BACKBONE.USE_PRM

        # VDT init
        num_depth = {'small': 8, 'base': 12,}[depth]
        num_heads = {'small': 8, 'base': 12,}[depth]
        mlp_ratio = {'small': 3., 'base': 4,}[depth]
        qkv_bias = {'small': False, 'base': True}[depth]
        qk_scale = {'small': 768 ** -0.5, 'base': None,}[depth]
        self.base = VisionTransformer_multiview(
            img_size=input_size, stride_size=stride_size,
            camera=cfg.INPUT.CAMERA,
            depth=num_depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_path_rate=drop_path_ratio, drop_rate=drop_ratio,
            attn_drop_rate=attn_drop_rate, inner_sub=inner_sub, local_feat=True
        )

        if pretrain:
            self.base.load_param(pretrain_path)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.image_pe = nn.Parameter(torch.zeros(1, self.base.num_patches, self.in_planes))
        trunc_normal_(self.image_pe, std=.02)
        self.out_token = nn.Parameter(torch.zeros(1, self.in_planes))
        trunc_normal_(self.out_token, std=.02)

        self.num_experts_per_type = cfg.MODEL.BACKBONE.EXPERTS
        self.top_k = cfg.MODEL.BACKBONE.TOP_K
        self.gtop_k = cfg.MODEL.BACKBONE.GRAPH_TOP_K
        self.share_experts = nn.ModuleList([
            MoEBlock(num_heads, self.in_planes, self.prompt_len) for _ in range(self.num_experts_per_type)
        ])
        self.sky_experts = nn.ModuleList([
            MoEBlock(num_heads, self.in_planes, self.prompt_len) for _ in range(self.num_experts_per_type)
        ])
        self.ground_experts = nn.ModuleList([
            MoEBlock(num_heads, self.in_planes, self.prompt_len) for _ in range(self.num_experts_per_type)
        ])
        self.share_router = Router(self.in_planes, self.num_experts_per_type, self.top_k)
        self.sky_router = Router(self.in_planes, self.num_experts_per_type, self.top_k)
        self.ground_router = Router(self.in_planes, self.num_experts_per_type, self.top_k)

        self.share_gcn = GraphConvolution(self.in_planes, class_num=self.prompt_len)
        self.view_gcn = GraphConvolution(self.in_planes, class_num=self.prompt_len)
        self.norm_share = nn.LayerNorm(self.in_planes)
        self.norm_view = nn.LayerNorm(self.in_planes)
        self.final_attn_token_to_image = OutAttenBlock(self.in_planes, num_heads)

    def _compute_expert_output(self, x, weights, indices, experts):
        output = torch.zeros(x.shape[0], self.prompt_len, x.shape[-1], device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_indices = indices[:, k]  # [B]
            expert_weights = weights[:, k]  # [B]

            for expert_idx in range(self.num_experts_per_type):
                mask = (expert_indices == expert_idx)  # [B]
                if mask.any():
                    selected_x = x[mask]  # [B_sel, 1, D]
                    expert_out = experts[expert_idx](selected_x)  # [B_sel, m, D]
                    output[mask] += expert_out * expert_weights[mask].view(-1, 1, 1)
        return output

    def forward(self, x, camera_id, view_id):
        B = x.shape[0]
        # VDT
        local_features = self.base(x, camera_id, view_id)
        local_feat = self.b1(local_features)
        global_features = local_feat[:, 0:1]
        view_features = local_feat[:, 1:2]
        local_feat = local_features[:, 2:]
        inv_features = global_features - view_features
        # strength_features = global_features + view_features

        # moe-prm
        moe_output = torch.zeros(B, self.prompt_len, self.in_planes, device=x.device, dtype=x.dtype)
        share_weights, share_indices, load_balance_loss_s = self.share_router(inv_features)
        share_output = self._compute_expert_output(inv_features, share_weights, share_indices, self.share_experts)
        sky_mask = (view_id == 0)
        ground_mask = (view_id == 1)
        if sky_mask.any():
            sky_weights, sky_indices, load_balance_loss_sky = self.sky_router(view_features[sky_mask])
            load_balance_loss_s += load_balance_loss_sky * sky_mask.sum() / B
            sky_output = self._compute_expert_output(view_features[sky_mask], sky_weights, sky_indices, self.sky_experts)
            # sky_concat = torch.cat([share_output[sky_mask], sky_output], dim=1)
            moe_output[sky_mask] = sky_output
        if ground_mask.any():
            ground_weights, ground_indices, load_balance_loss_ground = self.ground_router(view_features[ground_mask])
            load_balance_loss_s += load_balance_loss_ground * ground_mask.sum() / B
            ground_output = self._compute_expert_output(view_features[ground_mask], ground_weights, ground_indices, self.ground_experts)
            # ground_concat = torch.cat([share_output[ground_mask], ground_output], dim=1)
            moe_output[ground_mask] = ground_output

        local_feat_norm = F.normalize(local_feat, dim=-1)  # [B, N_local, D]
        inv_feat_norm = F.normalize(inv_features, dim=-1)  # [B, 1, D]
        view_feat_norm = F.normalize(view_features, dim=-1)  # [B, 1, D]
        sim_share = torch.matmul(inv_feat_norm, local_feat_norm.transpose(1, 2)).squeeze(1)
        sim_view = torch.matmul(view_feat_norm, local_feat_norm.transpose(1, 2)).squeeze(1)
        k = self.gtop_k
        topk_share = sim_share.topk(k, dim=-1).indices  # [B, k]
        topk_view = sim_view.topk(k, dim=-1).indices  # [B, k]

        node_cluster_local_share = torch.gather(
            local_feat, 1, topk_share.unsqueeze(-1).expand(-1, -1, local_feat.size(-1))
        ).unsqueeze(dim=1).repeat(1, share_output.shape[1], 1, 1)  # [B, N, k, D]
        node_cluster_local_view = torch.gather(
            local_feat, 1, topk_view.unsqueeze(-1).expand(-1, -1, local_feat.size(-1))
        ).unsqueeze(dim=1).repeat(1, share_output.shape[1], 1, 1)  # [B, N, k, D]

        with torch.no_grad():

            input_tokens_share = share_output.unsqueeze(dim=-2)
            feat_sp = torch.cat([input_tokens_share, node_cluster_local_share], dim=2)
            edge_sp = cal_edge_emb(feat_sp).detach()

            input_tokens_view = moe_output.unsqueeze(dim=-2)
            feat_vp = torch.cat([input_tokens_view, node_cluster_local_view], dim=2)
            edge_vp = cal_edge_emb(feat_vp).detach()

        graph_o_share = self.norm_share(self.share_gcn(feat_sp, edge_sp)) + share_output
        graph_o_view = self.norm_view(self.share_gcn(feat_vp, edge_vp)) + moe_output

        graph_o = torch.cat([self.out_token.unsqueeze(dim=0).repeat(B, 1, 1), graph_o_share, graph_o_view], dim=1)

        out_feat = self.final_attn_token_to_image(graph_o)

        return global_features.reshape(x.shape[0], -1, 1, 1), out_feat[:, 0].reshape(x.shape[0], -1, 1, 1), view_features.reshape(x.shape[0], -1, 1, 1), out_feat[:, 1:], load_balance_loss_s


@BACKBONE_REGISTRY.register()
def build_multiview_vit_backbone_ViSA(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off

    # fmt: on

    model = Vision_Transformer_ViSA(cfg)
    return model
