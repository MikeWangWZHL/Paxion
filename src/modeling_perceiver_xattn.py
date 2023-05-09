"""
adapt from https://github.com/lucidrains/flamingo-pytorch
"""


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(k_v_dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(k_v_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents, attn_mask=None, attn_guidance=None):
        """
        einstein notation
            b - batch
            t - time
            n - sequence
            d - dimension
        x: vision features (batch_size(b), num_video(t), num_tokens(n), k_v_dim(d))
        atten_guidance: a weighting for the x (i.e., raw_vision_features): (batch_size, num_video, num_tokens)
        latents: learnable query (b, t, num_latents, dim)
        """

        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, t, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # get key value based on x
        # kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        if attn_guidance is not None:
            attn_guidance = repeat(attn_guidance, "b t j -> b h t i j", h=h, i=sim.shape[-2])
            sim = sim * attn_guidance

        attn = sim.softmax(dim = -1)
        # print(attn)
        # print("attn.shape:", attn.shape)



        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        dim, # latent query hidden dimension
        k_v_dim, # vision features hidden dimension
        depth,
        dim_head = 64, # dimension for each cross-attention head
        heads = 8,
        num_latents = 64, # number of learnable latents
        # num_media_embeds = 4, # max number of vision feature sequences (e.g., images, videos)
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        # self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, k_v_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, k_v_dim = k_v_dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, latents=None, attn_mask=None, attn_guidance=None):
        # x: vision feature that provides key and value
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        if attn_guidance is not None and attn_guidance.ndim == 2:
            attn_guidance = rearrange(attn_guidance, 'b n -> b 1 n')

        ## assume that if input has multiple frames, pos_embed already included
        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times] # expected shape: (batch_size, num_video, num_tokens (patches*frms), dimension)

        if latents is None:
            # use unconditional latents
            latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])
        else:
            # use inputed conditional latents
            latents = repeat(latents, 'b 1 n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents, attn_mask=attn_mask, attn_guidance=attn_guidance) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

# TODO: query perceiver and knowledge perceiver
# TODO: patch-level attention guidance 

# cross attention
class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(k_v_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(k_v_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        k_v
    ):
        """
            x: providing features for query 
            k_v: providing features for key and value  
        """
        b, t, m, h = *k_v.shape[:3], self.heads
        
        x = self.norm_q(x)
        k_v = self.norm_kv(k_v)

        if len(k_v.shape) == 4:
            k_v = rearrange(k_v, 'b t m d -> b (t m) d')
        
        q = self.to_q(x)
        k, v = self.to_kv(k_v).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # for numerical stability
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# gated cross attention
class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, k_v_dim = k_v_dim, dim_head = dim_head, heads = heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        k_v
    ):
        x = self.attn(x, k_v) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

# cross attention
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, k_v_dim = k_v_dim, dim_head = dim_head, heads = heads)
        self.ff = FeedForward(dim, mult = ff_mult)

    def forward(
        self,
        x,
        k_v
    ):
        x = self.attn(x, k_v) + x
        x = self.ff(x) + x
        return x
