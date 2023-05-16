"""
    reference: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
"""
from PIL import Image
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import *
from lavis.models.base_model import BaseModel
from lavis.models.base_model import concat_all_gather
from lavis.models.base_model import MomentumDistilationMixin
from lavis.common.dist_utils import get_rank, get_world_size
import torch.distributed as dist
from copy import deepcopy

import logging
import json

from processors import *
import InternVideo
from einops import rearrange, repeat
from modeling_perceiver_xattn import (
    Perceiver,
    CrossAttentionBlock,
    GatedCrossAttentionBlock,
    FeedForward
)


logging.getLogger().setLevel(logging.INFO)
# torch.set_printoptions(profile="full")

## perceiver argument example
DEFAULT_QUERY_PERCEIVER_CONFIG = {
    "dim": 768, # latent query dim
    "k_v_dim": 768, # text_width
    "depth": 1,
    "dim_head": 64,
    "heads": 8,
    "num_latents": 16,
    "ff_mult": 2
}
DEFAULT_KNOWLEDGE_PERCEIVER_CONFIG = {
    "dim": 768, # latent query dim
    "k_v_dim": 1024, # vision_width
    "depth": 1,
    "dim_head": 64,
    "heads": 8,
    "num_latents": 16,
    "ff_mult": 2
}
DEFAULT_DYNAMIC_MODEL_XATTN_CONFIG={
    "dim": 768, # latent query dim
    "k_v_dim": 768, # latent query dim
    "dim_head": 64,
    "heads": 8,
    "ff_mult": 2
}

## for reformulated version of our model
DEFAULT_VISION_PERCEIVER_CONFIG = {
    "dim": 768, # latent query dim
    "k_v_dim": 1024, # vision_width
    "depth": 1,
    "dim_head": 64,
    "heads": 8,
    "num_latents": 16,
    "ff_mult": 2
}
DEFAULT_TEXT_PERCEIVER_CONFIG = {
    "dim": 768, # latent query dim
    "k_v_dim": 768, # text_width
    "depth": 1,
    "dim_head": 64,
    "heads": 8,
    "num_latents": 16,
    "ff_mult": 2
}

## default fuser config
DEFAULT_FUSER_XATTN_CONFIG={
    "dim": 768, # latent query dim
    "k_v_dim": 768, # latent query dim
    "dim_head": 64,
    "heads": 8,
    "ff_mult": 2
}

DEFAULT_FUSER_XATTN_CONFIG_V2={
    "dim": 768, # latent query dim
    "k_v_dim": 1024, # latent query dim
    "dim_head": 64,
    "heads": 8,
    "ff_mult": 2
}

## Objective names
VTC = "video_text_contrastive"
VAC = "video_action_contrastive"
ATM = "action_temporal_matching"


## function for loading pretrained ckpt
def load_from_pretrained(model, url_or_filename, key_mapping = {}):
    # key_mapping: mapping key names from checkpoint state_dict to model
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        print("load from path:",url_or_filename)
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    print("checkpoint keys:", checkpoint.keys())
    state_dict = checkpoint["model"]
    
    # renaming keys in state_dict to be compatible with model
    for key in list(state_dict.keys()):
        for bef, aft in key_mapping.items():
            if bef in key:
                state_dict[key.replace(bef, aft)] = state_dict[key]
                print("map key:", key, "-->", key.replace(bef, aft))
                del state_dict[key]

    # load keys that are in both model and checkpoint
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                print("del checkpoint param key that not matches the model param size:", key)
                del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)        
    # logging.info("Unexpected keys {}".format(msg.unexpected_keys))
    logging.info("Missing keys {}".format(msg.missing_keys))
    logging.info("load checkpoint from %s" % url_or_filename)

    return msg

## KnowledgePatcher base model
class KnowledgePatcherBase(BaseModel):
    def __init__(
        self,
        vl_backbone
    ):
        super().__init__()
        self.vl_backbone = vl_backbone
        self._freeze_backbone()
        
    def _freeze_backbone(self):
        for param in self.vl_backbone.parameters():
            param.requires_grad = False

    def tokenize(self, text, **kwargs):
        raise NotImplementedError('tokenize method are required to be implemented')

    def encode_text(self, text, return_all_feats=True, masked_indices=None):
        raise NotImplementedError('encode_text method not implemented')
        
    def encode_video(self, video, raw_text_embeddings=None, return_all_feats=True, masked_indices=None):
        raise NotImplementedError('encode_video method not implemented')

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        raise NotImplementedError('encode_text_backbone method not implemented')

    def encode_video_backbone(self, video, return_all_feats=False, masked_indices=None):
        raise NotImplementedError('encode_video_backbone method not implemented')
    
    def forward_backbone(self, x):
        return self.vl_backbone(x)

    def forward(self, x):
        raise NotImplementedError('forward method not implemented')

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        rel_path = cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]
        src_root = os.path.dirname(os.path.abspath(__file__))
        model_default_config_path = os.path.join(src_root, rel_path)
        return model_default_config_path

### --------- Patch & Fuse: InternVideo --------- ###
@registry.register_model("patch_and_fuse_internvideo")
class PaxionInternVideo(KnowledgePatcherBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "InternVideo-MM-L-14": "configs/models/patch_and_fuse_intern_video.yaml",
    }
    def __init__(self, 
        vl_backbone,
        vision_perceiver_config = DEFAULT_VISION_PERCEIVER_CONFIG,
        text_perceiver_config = DEFAULT_TEXT_PERCEIVER_CONFIG,
        knowledge_fuser_config = DEFAULT_FUSER_XATTN_CONFIG,
        objectives = [
            VTC,
            VAC,
            ATM,
        ],
        loss_weighting = [
            1.0,
            1.0,
            1.0,
        ], # weighting of the provided objectives 
        if_use_attn_guidance = False,
        if_use_dual_perceiver = False,
        if_add_temporal_emebdding  = False,
        state_change_filtering_for_FDM = False,
        temp_emb_drop_out = 0,
        num_frms = 8,
        if_as_knowledge_fuser = False,
        knowledge_fuser_type = "xattn", # "xattn", "gated_xattn", "xattn_v2", "side_tuning"
        train_knowledge_fuser_jointly = False,
        if_pooling_perceiver_features = False
    ):
        super().__init__(vl_backbone)

        assert len(objectives) == len(loss_weighting)
        self.loss_weighting = {objectives[i]:loss_weighting[i] for i in range(len(objectives))}
        self.if_use_attn_guidance = if_use_attn_guidance
        self.if_use_dual_perceiver = if_use_dual_perceiver
        self.if_add_temporal_emebdding = if_add_temporal_emebdding
        self.if_as_knowledge_fuser = if_as_knowledge_fuser
        self.knowledge_fuser_type = knowledge_fuser_type
        self.state_change_filtering_for_FDM = state_change_filtering_for_FDM
        self.if_pooling_perceiver_features = if_pooling_perceiver_features
        
        # main layers
        self.vision_perceiver = Perceiver(**vision_perceiver_config)
        if if_use_dual_perceiver:
            self.text_perceiver = Perceiver(**text_perceiver_config)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        if self.if_add_temporal_emebdding:
            # temporal embedding for frames
            self.temporal_embedding = nn.Parameter(torch.randn(num_frms, vision_perceiver_config['k_v_dim']))
            if temp_emb_drop_out > 0:
                self.temp_emb_drop_out = nn.Dropout(p=temp_emb_drop_out)
            else:
                self.temp_emb_drop_out = None
        
        
        if self.if_as_knowledge_fuser:
            if knowledge_fuser_type in ['xattn','xattn_v2']:
                self.knowledge_fuser = CrossAttentionBlock(**knowledge_fuser_config)
            elif knowledge_fuser_type == 'gated_xattn':
                self.knowledge_fuser = GatedCrossAttentionBlock(**knowledge_fuser_config)
            elif knowledge_fuser_type == 'side_tuning': 
                # following: https://github.com/jozhang97/side-tuning/blob/master/tlkit/models/sidetune_architecture.py
                init_value = 0.0 # init alpha to be 0 => initially 50%/50% of the knowledge patcher (side network) and the backbone (base network)
                self.alpha = nn.Parameter(torch.tensor(init_value))
                train_knowledge_fuser_jointly = True # make knowledge patcher trainable as the side model

            if not train_knowledge_fuser_jointly:
                # freeze trained vision perceiver
                self._freeze_knowledge_patcher_before_fuse()

    def _freeze_knowledge_patcher_before_fuse(self):
        # freeze everything
        for param in self.parameters():
            param.requires_grad = False
        # unfreeze knowledge fuser
        for param in self.knowledge_fuser.parameters():
            param.requires_grad = True
        self.temp.requires_grad = True
        # NOTE: unfreeze text perceiver if exist
        if self.if_use_dual_perceiver:
            for param in self.text_perceiver.parameters():
                param.requires_grad = True

    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return InternVideo.tokenize(text, truncate=True, return_special_tokens_mask=False)

    def encode_text(self, 
        text,
        video_embeddings=None,
        attn_guidance=None,
        masked_indices=None
    ):
        raw_text_feat, raw_text_embeddings = self.encode_text_backbone(
            text, 
            return_all_feats=True, 
            masked_indices=masked_indices
        ) # raw_text_feat: (B, dim_t); raw_text_embeddings: (B, L, dim_t)

        if self.if_use_attn_guidance and attn_guidance is None:
            attn_guidance = self._get_attn_guidance(video_embeddings, raw_text_feat)
        else:
            attn_guidance = None

        if self.if_use_dual_perceiver:
            percevier_textual_embeddings = self.text_perceiver(
                raw_text_embeddings, 
                latents=None, 
                attn_mask=masked_indices, 
                attn_guidance=attn_guidance
            ) # percevier_textual_embeddings: (B, 1, num_latents, dim_latents)
            return raw_text_feat, percevier_textual_embeddings[:,0,:,:] # (B, dim_v_projected) (B, num_latents, dim_latents)
        else:
            return raw_text_feat, raw_text_embeddings # (B, dim_t) (B, L, dim_t)
    
    def encode_video(self, 
        video,
        raw_visual_feat = None,
        raw_visual_embeddings = None,
        text_embeddings=None,
        attn_guidance=None,
        masked_indices=None
    ):
        if raw_visual_embeddings is None:
            assert video is not None
            raw_visual_feat, raw_visual_embeddings = self.encode_video_backbone(
                video, 
                return_all_feats=True, 
                masked_indices=masked_indices
            ) # raw_visual_feat: (B, dim_v_projected);  raw_visual_embeddings: (P, B, num_frm, dim_v)
            raw_visual_embeddings = rearrange(raw_visual_embeddings, "p b f d -> b f p d") # (B, num_frm, num_patches, dim_v)
        else:
            assert raw_visual_feat is not None
        
        if self.if_add_temporal_emebdding:
            temporal_embedding = repeat(self.temporal_embedding, "f d -> f p d", p = raw_visual_embeddings.shape[2])
            if self.temp_emb_drop_out is not None:
                temporal_embedding = self.temp_emb_drop_out(temporal_embedding)
            raw_visual_embeddings_w_temp_emb = raw_visual_embeddings + temporal_embedding
        else:  
            raw_visual_embeddings_w_temp_emb = raw_visual_embeddings

        if self.if_use_attn_guidance and attn_guidance is None:
            attn_guidance = self._get_attn_guidance(raw_visual_embeddings, text_embeddings)
        else:
            attn_guidance = None

        raw_visual_embeddings = rearrange(raw_visual_embeddings, "b f p d -> b (f p) d") # (B, (num_frm*num_patches), dim_v)
        raw_visual_embeddings_w_temp_emb = rearrange(raw_visual_embeddings_w_temp_emb, "b f p d -> b (f p) d") # (B, (num_frm*num_patches), dim_v)

        percevier_visual_embeddings = self.vision_perceiver(
            raw_visual_embeddings_w_temp_emb, 
            latents=None, 
            attn_mask=masked_indices, 
            attn_guidance=attn_guidance
        )[:,0,:,:] # percevier_visual_embeddings: (B, num_latents, dim_latents)

        if self.if_as_knowledge_fuser:
            if self.knowledge_fuser_type in ["xattn_v2"]:                
                fused_visual_feat = self.knowledge_fuser(
                    percevier_visual_embeddings, # (B, num_latents, dim_latents)
                    raw_visual_embeddings # B, (num_frm*num_patches), dim_v
                )
            elif self.knowledge_fuser_type in ["side_tuning"]:
                alpha_squashed = torch.sigmoid(self.alpha)
                side_visual_feature = F.normalize(percevier_visual_embeddings.mean(1), dim=-1) # (B, dim_latents)
                fused_visual_feat = alpha_squashed * F.normalize(raw_visual_feat, dim=-1) + (1 - alpha_squashed) * side_visual_feature # (B, dim_latents)
            else:
                fused_visual_feat = self.knowledge_fuser(
                    rearrange(raw_visual_feat, "b d -> b 1 d"),
                    percevier_visual_embeddings # (B, num_latents, dim_latents)
                )[:,0,:] # (B, dim_latents)
            return raw_visual_feat, fused_visual_feat
        else:
            if self.if_pooling_perceiver_features:
                assert percevier_visual_embeddings.dim() == 3
                percevier_visual_embeddings = percevier_visual_embeddings.mean(1) # (B, Q, D) -> (B, D) 
            return raw_visual_feat, percevier_visual_embeddings # (B, dim_v_projected), (B, num_latents, dim_latents) or (B, dim_latents) if pooling

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        return self.vl_backbone.encode_text(text, return_all_feats=return_all_feats, masked_indices=masked_indices)
    
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        # set num_frm in backbone visiontransformer
        num_frm = video.shape[1]
        self.vl_backbone.visual.transformer.T = num_frm

        video = rearrange(video, 'b m c h w -> b c m h w') # InternVideo expect input to be (B, C, num_frm, H, W)
        return self.vl_backbone.encode_video(video, return_all_feats=return_all_feats, mode=mode)

    def _get_attn_guidance(self, visual_embeddings, text_embeddings):
        raise NotImplementedError

    def _get_contrastive_loss(
        self, 
        perceiver_v_feat, 
        text_feat, 
        perceiver_v_feat_all, 
        text_feat_all, 
        batch_size, 
        device
    ):
        assert perceiver_v_feat.dim() == perceiver_v_feat_all.dim() == 3
        if text_feat.dim() == 2:
            assert text_feat_all.dim() == 2 # (batch_size*num_gpu, latent_dim)
            # visual queries - single text feat similarity 
            sim_q2t = torch.matmul(
                perceiver_v_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                text_feat_all.unsqueeze(-1) # (batch_size*num_gpu, latent_dim, 1)
            ).squeeze(-1) # (batch_size, batch_size*num_gpu, num_latents)

            # video-text similarity: aggregate across all visual query tokens
            sim_v2t, _ = sim_q2t.max(-1) # value, indice
            sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)

            # text-query similarity: [batch_size, batch_size*num_gpu, num_latents]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), # (batch_size, 1, 1, latent_dim,)
                perceiver_v_feat_all.permute(0, 2, 1)  # (batch_size*num_gpu, latent_dim, num_latents)
            ).squeeze(-2)
            
            # text-video similarity: aggregate across all query tokens
            sim_t2v, _ = sim_t2q.max(-1)
            sim_t2v = sim_t2v / self.temp  # (batch_size, batch_size*num_gpu)

        elif text_feat.dim() == 3:
            assert text_feat_all.dim() == 3 # (batch_size*num_gpu, num_latents, latent_dim)
            # visual queries - textual queries similarity
            sim_q2t = torch.matmul(
                perceiver_v_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                rearrange(text_feat_all, "c l d -> c d l") # (batch_size*num_gpu, latent_dim, num_latents)
            ) # (batch_size, batch_size*num_gpu, num_latents, num_latents)
            # take mean across target tokens and then take max across all input tokens
            sim_v2t, _ = sim_q2t.mean(dim=-1).max(dim=-1) # (batch_size, batch_size*num_gpu)
            sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)

            # text queries - visual queries similarity
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                rearrange(perceiver_v_feat_all, "c l d -> c d l") # (batch_size*num_gpu, latent_dim, num_latents)
            ) # (batch_size, batch_size*num_gpu, num_latents, num_latents)
            # take mean across target tokens and then take max across all input tokens
            sim_t2v, _ = sim_t2q.mean(dim=-1).max(dim=-1) # (batch_size, batch_size*num_gpu)
            sim_t2v = sim_t2v / self.temp # (batch_size, batch_size*num_gpu)

        rank = get_rank()
        world_size = get_world_size()
        bs = batch_size
        t_all_bs = sim_v2t.shape[1]//world_size
        v_all_bs = sim_t2v.shape[1]//world_size
        v2t_targets = torch.linspace(rank * t_all_bs, rank * t_all_bs + bs - 1, bs, dtype=int).to(device)
        t2v_targets = torch.linspace(rank * v_all_bs, rank * v_all_bs + bs - 1, bs, dtype=int).to(device)

        loss_vtc = (
            F.cross_entropy(sim_v2t, v2t_targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2v, t2v_targets, label_smoothing=0.1)
        ) / 2

        return loss_vtc

    def _get_atm_loss(self, 
        perceiver_v_feat,
        perceiver_v_feat_shuffled,
        perceiver_v_feat_reversed,
        text_feat
    ):
        if perceiver_v_feat_shuffled is not None:
            perceiver_v_feat_concat = torch.concat(
                [
                    perceiver_v_feat.unsqueeze(1), 
                    perceiver_v_feat_shuffled.unsqueeze(1),
                    perceiver_v_feat_reversed.unsqueeze(1)
                ], dim = 1) # (B,3,Q,D)
        else:
            perceiver_v_feat_concat = torch.concat(
                [
                    perceiver_v_feat.unsqueeze(1), 
                    perceiver_v_feat_reversed.unsqueeze(1)
                ], dim = 1) # (B,2,Q,D)

        if text_feat.dim() == 2: # (B, D)
            # single text feat - visual query tokens similarity 
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), # (batch_size, 1, 1, latent_dim)
                rearrange(perceiver_v_feat_concat, "b l q d -> b q d l")  # (batch_size, num_latents, latent_dim, 3)
            ).squeeze(-2) # (batch_size, num_latants, 3)

            # max across all visual query tokens
            logits, _ = sim_t2q.max(1) # value, indice

            # # mean acrsso all visual query tokens
            # logits = sim_t2q.mean(1)

            logits = logits / self.temp # (batch_size, 3)

        elif text_feat.dim() == 3: # (B, Q, D)
            # text query tokens - visual query tokens similarity 
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1), # (batch_size, 1, num_latents_t, latent_dim)
                rearrange(perceiver_v_feat_concat, "b l q d -> b q d l")  # (batch_size, num_latents_v, latent_dim, 3)
            ) # (batch_size, num_latents_v, num_latents_t, 3)

            # mean across all textual tokens and max across all visual query tokens
            logits, _ = sim_t2q.mean(2).max(1) # value, indice

            # # mean across all textual tokens and mean across all visual query tokens
            # logits = sim_t2q.mean(2).mean(1)

            logits = logits / self.temp # (batch_size, 3)

        # print("forward dm logits first 4 samples:",logits[:4,:])
        targets = torch.zeros(logits.shape[0],dtype=int).to(logits.device)
        loss_atm= F.cross_entropy(logits, targets, label_smoothing=0.1)
        return loss_atm
    
    def _get_contrastive_loss_fuser(
        self, 
        visual_feat, 
        text_feat, 
        visual_feat_all, 
        text_feat_all, 
        batch_size, 
        device
    ):
        
        if text_feat.dim() == 2:

            if visual_feat.dim() == 2: # (B, D)  
                sim_v2t = (visual_feat @ text_feat_all.T) / self.temp # (batch_size, batch_size*num_gpu)
                sim_t2v = (text_feat @ visual_feat_all.T) / self.temp # (batch_size, batch_size*num_gpu)
            
            elif visual_feat.dim() == 3: # (B, Q, D)
                # query-text similarity [batch_size, batch_size*num_gpu, num_query_tokens]
                sim_q2t = torch.matmul(
                    visual_feat.unsqueeze(1), # (batch_size, 1, num_query_tokens, latent_dim)  
                    text_feat_all.unsqueeze(-1) # (batch_size*num_gpu, latent_dim, 1)
                ).squeeze(-1)
                # video-text similarity: aggregate across all query tokens
                sim_v2t, _ = sim_q2t.max(-1) # value, indice
                sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)

                # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
                sim_t2q = torch.matmul(
                    text_feat.unsqueeze(1).unsqueeze(1), # (batch_size, 1, 1, latent_dim,)
                    rearrange(visual_feat_all, "c q d -> c d q")  # (batch_size*num_gpu, latent_dim, num_query_tokens)
                ).squeeze(-2)
                # text-video similarity: aggregate across all query tokens
                sim_t2v, _ = sim_t2q.max(-1)
                sim_t2v = sim_t2v / self.temp  # [batch_size, batch_size*num_gpu]

        elif text_feat.dim() == 3:

            if visual_feat.dim() == 2:
                # visual feat - textual queries similarity
                sim_v2q = torch.matmul(
                    visual_feat.unsqueeze(1).unsqueeze(1), # (batch_size, 1, 1, latent_dim)  
                    rearrange(text_feat_all, "c l d -> c d l") # (batch_size*num_gpu, latent_dim, num_latents)
                ).squeeze(-2) # (batch_size, batch_size*num_gpu, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_v2t, _ = sim_v2q.max(-1) # (batch_size, batch_size*num_gpu)
                sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)

                # text queries - visual queries similarity
                sim_q2v = torch.matmul(
                    text_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                    visual_feat_all.unsqueeze(-1) # (batch_size*num_gpu, latent_dim, 1)
                ).squeeze(-1) # (batch_size, batch_size*num_gpu, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_t2v, _ = sim_q2v.max(dim=-1) # (batch_size, batch_size*num_gpu)
                sim_t2v = sim_t2v / self.temp # (batch_size, batch_size*num_gpu)
            
            elif visual_feat.dim() == 3:
                # visual feat - textual queries similarity
                sim_v2q = torch.matmul(
                    visual_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                    rearrange(text_feat_all, "c l d -> c d l") # (batch_size*num_gpu, latent_dim, num_latents)
                ) # (batch_size, batch_size*num_gpu, num_latents, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_v2t, _ = sim_v2q.mean(dim=-1).max(dim=-1) # (batch_size, batch_size*num_gpu)
                sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)

                # text queries - visual queries similarity
                sim_q2v = torch.matmul(
                    text_feat.unsqueeze(1), # (batch_size, 1, num_latents, latent_dim)  
                    rearrange(visual_feat_all, "c l d -> c d l") # (batch_size*num_gpu, latent_dim, num_latents)
                ).squeeze(-1) # (batch_size, batch_size*num_gpu, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_t2v, _ = sim_q2v.mean(dim=-1).max(dim=-1) # (batch_size, batch_size*num_gpu)
                sim_t2v = sim_t2v / self.temp # (batch_size, batch_size*num_gpu)

        rank = get_rank()
        bs = batch_size
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(device)
        
        # TODO: incorporate loss_weighting for inverse dynamic part
        loss_vtc = (
            F.cross_entropy(sim_v2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
        ) / 2

        return loss_vtc

    def forward(self, samples):
        video_input = samples["video_input"] # (B, num_frm, C, H, W) 
        text_input = samples["text_input"]
        batch_size = video_input.size(0)

        ### get backbone features
        text_input_tokenized = self.tokenize(text_input).to(video_input.device)
        raw_text_feat, raw_text_embeddings = self.encode_text_backbone(
            text_input_tokenized, return_all_feats=True
        ) # (B, dim_t_projected) (B, L, dim_t)
        raw_visual_feat, raw_visual_embeddings = self.encode_video_backbone(
            video_input, return_all_feats=True
        ) # (B, dim_v_projected) (P, B, num_frms, dim_v)
        raw_visual_embeddings = rearrange(
            raw_visual_embeddings, "p b f d -> b f p d"
        ) # (B, num_frm, num_patches, dim_v)
        
        ### get attention guidance if specified
        if self.if_use_attn_guidance:
            attn_guidance = self._get_attn_guidance(raw_visual_embeddings, raw_text_embeddings)
        else:
            attn_guidance = None

        ### get perceiver features - original
        if self.if_add_temporal_emebdding:
            temporal_embedding = repeat(self.temporal_embedding, "f d -> f p d", p = raw_visual_embeddings.shape[2])
            if self.temp_emb_drop_out is not None:
                temporal_embedding = self.temp_emb_drop_out(temporal_embedding)
            raw_visual_embeddings_w_temp_emb = raw_visual_embeddings + temporal_embedding
        else:
            raw_visual_embeddings_w_temp_emb = raw_visual_embeddings

        percevier_visual_embeddings = self.vision_perceiver(
            rearrange(raw_visual_embeddings_w_temp_emb, "b f p d -> b (f p) d"), # (B, (num_frm*num_patches), dim_v)
            latents=None, 
            attn_mask=None,
            attn_guidance=attn_guidance
        )[:,0,:,:] # (B, num_latents, dim_latents)

        if self.if_use_dual_perceiver:
            percevier_textual_embeddings = self.text_perceiver(
                raw_text_embeddings, 
                latents=None, 
                attn_mask=None, 
                attn_guidance=attn_guidance
            )[:,0,:,:] # (B, num_latents, dim_latents)
            text_feat = F.normalize(percevier_textual_embeddings, dim=-1)
        else:
            text_feat = F.normalize(raw_text_feat, dim=-1)

        losses = {}
        if self.if_as_knowledge_fuser:
            #### training knowledge fuser ####
            if self.knowledge_fuser_type in ["xattn_v2"]: 
                fused_visual_feat = self.knowledge_fuser(
                    percevier_visual_embeddings, # (B, num_latents, dim_latents)
                    rearrange(raw_visual_embeddings, "b f p d -> b (f p) d") # B, (num_frm*num_patches), dim_v
                )
            elif self.knowledge_fuser_type in ["side_tuning"]:
                alpha_squashed = torch.sigmoid(self.alpha)
                side_visual_feature = F.normalize(percevier_visual_embeddings.mean(1), dim=-1) # (B, dim_latents)
                fused_visual_feat = alpha_squashed * F.normalize(raw_visual_feat, dim=-1) + (1 - alpha_squashed) * side_visual_feature # (B, dim_latents)
            else:
                fused_visual_feat = self.knowledge_fuser(
                    rearrange(raw_visual_feat, "b d -> b 1 d"),
                    percevier_visual_embeddings
                )[:,0,:] # (B, D)
            
            fused_visual_feat = F.normalize(fused_visual_feat, dim=-1)
            fused_visual_feat_all = concat_all_gather(fused_visual_feat)
            text_feat_all = concat_all_gather(text_feat)
            loss_vtc = self._get_contrastive_loss_fuser(
                fused_visual_feat, 
                text_feat, 
                fused_visual_feat_all, 
                text_feat_all, 
                batch_size, 
                video_input.device
            )
            losses[VTC] = loss_vtc * self.loss_weighting[VTC]
        else: 
            #### training knowledge patcher ####
            ### ============== Video-text Contrastive =================== ###
            perceiver_v_feat = F.normalize(percevier_visual_embeddings, dim=-1)
            if VTC in self.loss_weighting:
                if VAC in self.loss_weighting:
                    ### ============== VTC + Video-Action Contrastive (VAC) =================== ###
                    ## encode action antonym text
                    antonym_text_input = samples["action_antonym_text_input"]
                    antonym_text_input_tokenized = self.tokenize(antonym_text_input).to(video_input.device)
                    raw_text_feat_antonym, raw_text_embeddings_antonym = self.encode_text_backbone(
                        antonym_text_input_tokenized, return_all_feats=True
                    ) # (B, dim_t) (B, L, dim_t)

                    if self.if_use_attn_guidance:
                        attn_guidance_antonym = self._get_attn_guidance(raw_visual_embeddings, raw_text_embeddings_antonym)
                    else:
                        attn_guidance_antonym = None
                    if self.if_use_dual_perceiver:
                        percevier_textual_embeddings_antonym = self.text_perceiver(
                            raw_text_embeddings_antonym, 
                            latents=None, 
                            attn_mask=None, 
                            attn_guidance=attn_guidance_antonym
                        )[:,0,:,:] # (B, num_latents, dim_latents)
                        text_feat_antonym = F.normalize(percevier_textual_embeddings_antonym, dim=-1)
                    else:
                        text_feat_antonym = F.normalize(raw_text_feat_antonym, dim=-1)

                    text_feat_concat = torch.cat([text_feat, text_feat_antonym], dim=0) # (2*batch_size, latent_dim) || (2*batch_size, num_latents, latent_dim)
                    text_feat_all = concat_all_gather(text_feat_concat)
                else:
                    ### ============== VTC-only =================== ###
                    text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, latent_dim]

                perceiver_v_feat_all = concat_all_gather(perceiver_v_feat)  # [batch_size*num_gpu, num_query_tokens, latent_dim]
                loss_vtc = self._get_contrastive_loss(
                    perceiver_v_feat, 
                    text_feat, 
                    perceiver_v_feat_all, 
                    text_feat_all, 
                    batch_size, 
                    video_input.device
                )
                losses[VTC] = loss_vtc * self.loss_weighting[VTC]

            if ATM in self.loss_weighting:
                ## encode video candidates

                # perm = torch.randperm(video_input.shape[1]) # get random indices of the frame axis
                # shuffled_video = video_input[:, perm, :, :, :] # (B, num_frm, C, H, W) 
                if self.state_change_filtering_for_FDM:
                    # use only the videos with strong state changes for FDM
                    flag_tensor = samples["state_change_saliency_flag"]
                    filter_indices = torch.nonzero(flag_tensor).squeeze().to(video_input.device)
                    filtered_video_input = torch.index_select(video_input,dim=0,index=filter_indices)
                    if filtered_video_input.shape[0] == 0:
                        reversed_video = None
                    else:
                        reversed_video = torch.flip(filtered_video_input, dims=[1]) # (B', num_frm, C, H, W) 
                    perceiver_v_feat_filtered = torch.index_select(perceiver_v_feat,dim=0,index=filter_indices) #(B', Q, D)
                    text_feat_filtered = torch.index_select(text_feat,dim=0,index=filter_indices) # (B', D)
                else:
                    reversed_video = torch.flip(video_input, dims=[1]) # (B, num_frm, C, H, W) 
                    perceiver_v_feat_filtered = perceiver_v_feat
                    text_feat_filtered = text_feat

                if reversed_video is not None:
                    # raw_visual_feat_shuffled, percevier_visual_embeddings_shuffled = self.encode_video(shuffled_video)
                    raw_visual_feat_reversed, percevier_visual_embeddings_reversed = self.encode_video(reversed_video)

                    # perceiver_v_feat_shuffled = F.normalize(percevier_visual_embeddings_shuffled, dim=-1)
                    perceiver_v_feat_shuffled_filtered = None # NOTE: not using shuffled for now
                    perceiver_v_feat_reversed_filtered = F.normalize(percevier_visual_embeddings_reversed, dim=-1)
                    assert perceiver_v_feat_filtered.shape[0] == perceiver_v_feat_reversed_filtered.shape[0] == text_feat_filtered.shape[0]
                    loss_atm = self._get_atm_loss(
                        perceiver_v_feat_filtered, 
                        perceiver_v_feat_shuffled_filtered, 
                        perceiver_v_feat_reversed_filtered, 
                        text_feat_filtered
                    )
                    losses[ATM] = loss_atm * self.loss_weighting[ATM]

        # aggregate loss
        total_loss = []
        for key, loss in losses.items():
            total_loss.append(loss)
        total_loss = torch.sum(torch.stack(total_loss))
        return total_loss, losses

    @classmethod
    def _load_backbone(cls, cfg):
        """ load pretrained Internvideo backbone """
        backbone_pretrained_ckpt = cfg.get("backbone_pretrained", None)
        assert backbone_pretrained_ckpt != None
        vl_backbone = InternVideo.load_model(backbone_pretrained_ckpt)
        logging.info("Loaded backbone pretrained weights from {}".format(backbone_pretrained_ckpt))
        return vl_backbone

    @classmethod
    def from_config(cls, cfg=None):
        logging.info("Model config: {}".format(cfg))
        vl_backbone = cls._load_backbone(cfg)

        vision_perceiver_config = cfg.get("vision_perceiver_config", DEFAULT_KNOWLEDGE_PERCEIVER_CONFIG)
        text_perceiver_config = cfg.get("text_perceiver_config", DEFAULT_KNOWLEDGE_PERCEIVER_CONFIG)
        objectives = cfg.get("objectives", [VTC,VAC,ATM])
        loss_weighting = cfg.get("loss_weighting", [1.0,1.0,1.0])
        if_use_attn_guidance = cfg.get("if_use_attn_guidance", False)
        if_use_dual_perceiver = cfg.get("if_use_dual_perceiver", False)
        if_add_temporal_emebdding = cfg.get("if_add_temporal_emebdding", False)
        num_frms = cfg.get("num_frms", 8)
        temp_emb_drop_out = cfg.get("temp_emb_drop_out", 0)
        if_as_knowledge_fuser = cfg.get("if_as_knowledge_fuser", False)
        knowledge_fuser_type = cfg.get("knowledge_fuser_type", "xattn") # ["xattn", "gated_xattn", "xattn_v2"]
        train_knowledge_fuser_jointly = cfg.get("train_knowledge_fuser_jointly", False) # ["xattn", "gated_xattn", "xattn_v2"]
        if knowledge_fuser_type in ["xattn_v2"]:
            knowledge_fuser_config = cfg.get("knowledge_fuser_config", DEFAULT_FUSER_XATTN_CONFIG_V2)
        else:
            knowledge_fuser_config = cfg.get("knowledge_fuser_config", DEFAULT_FUSER_XATTN_CONFIG)
        
        state_change_filtering_for_FDM = cfg.get("state_change_filtering_for_FDM", False)
        if_pooling_perceiver_features = cfg.get("if_pooling_perceiver_features", False)


        print("vision_perceiver_config:",vision_perceiver_config)
        print("text_perceiver_config:",text_perceiver_config)
        print("knowledge_fuser_config:",knowledge_fuser_config)
        print("objectives:",objectives)
        print("loss_weighting:",loss_weighting)
        print("if_use_attn_guidance:",if_use_attn_guidance)
        print("if_use_dual_perceiver:",if_use_dual_perceiver)
        print("if_as_knowledge_fuser:",if_as_knowledge_fuser)
        print("train_knowledge_fuser_jointly:",train_knowledge_fuser_jointly)
        print("if_add_temporal_emebdding:",if_add_temporal_emebdding, "| num_frms:", num_frms)
        print("knowledge_fuser_type:",knowledge_fuser_type)
        print("temp_emb_drop_out:",temp_emb_drop_out)
        print("state_change_filtering_for_FDM:",state_change_filtering_for_FDM)
        print("if_pooling_perceiver_features:",if_pooling_perceiver_features)
        
        model = cls(
            vl_backbone,
            vision_perceiver_config=vision_perceiver_config,
            text_perceiver_config=text_perceiver_config,
            knowledge_fuser_config=knowledge_fuser_config,
            objectives=objectives,
            loss_weighting=loss_weighting,
            if_use_attn_guidance=if_use_attn_guidance,
            if_use_dual_perceiver=if_use_dual_perceiver,
            if_add_temporal_emebdding=if_add_temporal_emebdding,
            state_change_filtering_for_FDM=state_change_filtering_for_FDM,
            temp_emb_drop_out=temp_emb_drop_out,
            num_frms=num_frms,
            if_as_knowledge_fuser=if_as_knowledge_fuser,
            knowledge_fuser_type=knowledge_fuser_type,
            train_knowledge_fuser_jointly=train_knowledge_fuser_jointly,
            if_pooling_perceiver_features=if_pooling_perceiver_features,
        )

        # load trained ckpt if specified
        pretrained_ckpt = cfg.get("pretrained", None)
        load_pretrained = cfg.get("load_pretrained", True)
        if (pretrained_ckpt is not None) and load_pretrained:
            logging.info("Loading pretrained weights...")
            msg = load_from_pretrained(model, url_or_filename=pretrained_ckpt, key_mapping={"knowledge_perceiver":"vision_perceiver"})
        else:
            logging.info("No additional pretrained weights are loaded.")

        return model

### --------- Patch & Fuse: ClipVIP --------- ###
import sys
from transformers import CLIPTokenizerFast
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "ClipViP"))
import torch
from ClipViP.src.modeling.VidCLIP import VidCLIP
from ClipViP.src.configs.config import shared_configs
from easydict import EasyDict as edict

@registry.register_model("patch_and_fuse_clipvip")
class PaxionClipVIP(PaxionInternVideo):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vip_base_32": "configs/models/patch_and_fuse_clip_vip.yaml",
    }
    def __init__(
        self,
        vl_backbone,
        **kwargs
    ):
        super().__init__(vl_backbone, **kwargs)
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return self.tokenizer(text, padding=True, return_tensors="pt")

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        clipmodel = self.vl_backbone.clipmodel
        text_outputs = clipmodel.text_model(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask'],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        pooled_output = text_outputs.pooler_output # (B, dim_t)
        text_embeddings = text_outputs.last_hidden_state # (B, L, dim_t)
        text_features = clipmodel.text_projection(pooled_output) 
        # noramize
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if return_all_feats:
            return text_features, text_embeddings
        return text_features
    
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        #TODO: debug this
        B, num_frm, C, H, W = video.shape
        clipmodel = self.vl_backbone.clipmodel
        vision_outputs = clipmodel.vision_model(
            pixel_values=video,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        pooled_output = vision_outputs.pooler_output  # (B, dim_v)
        video_embeddings = vision_outputs.last_hidden_state[:,4:,:] # (B, num_frm*P, dim_v); the first 4 elements are the cls tokens
        video_embeddings = rearrange(video_embeddings, 'b (n p) d -> p b n d', n=num_frm)
        video_features = clipmodel.visual_projection(pooled_output)
        # noramize
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        if return_all_feats:
            return video_features, video_embeddings
        return video_features
    
    @classmethod
    def _load_backbone(cls, cfg):
        """ load pretrained CLIP-Vip backbone """
        
        def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
            """operated in-place, no need to return `model`"""

            if isinstance(loaded_state_dict_or_path, str):
                loaded_state_dict = torch.load(
                    loaded_state_dict_or_path, map_location="cpu")
            else:
                loaded_state_dict = loaded_state_dict_or_path
            model_keys = set([k for k in list(model.state_dict().keys())])
            load_keys = set(loaded_state_dict.keys())

            toload = {}
            mismatched_shape_keys = []
            for k in model_keys:
                if k in load_keys:
                    if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                        mismatched_shape_keys.append(k)
                    else:
                        toload[k] = loaded_state_dict[k]

            print("You can ignore the keys with `num_batches_tracked` or from task heads")
            print("Keys in loaded but not in model:")
            diff_keys = load_keys.difference(model_keys)
            print(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
            print("Keys in model but not in loaded:")
            diff_keys = model_keys.difference(load_keys)
            print(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
            print("Keys in model and loaded, but shape mismatched:")
            print(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
            model.load_state_dict(toload, strict=False)
        
        def setup_model(cfg):
            print("Setup model...")
            
            model = VidCLIP(cfg)

            if cfg.e2e_weights_path:
                print(f"Loading e2e weights from {cfg.e2e_weights_path}")
                load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
            
            if hasattr(cfg, "overload_logit_scale"):
                model.overload_logit_scale(cfg.overload_logit_scale)
            
            print("Setup model done!")
            return model

        backbone_config_json = cfg.get("backbone_config_json", None)
        assert backbone_config_json != None
        # parsed_args = shared_configs.get_pretraining_args()
        # cfg = load_config_json_to_args(parsed_args, backbone_config_json)
        # vl_backbone = setup_model(cfg)
        cfg = edict(json.load(open(backbone_config_json)))
        # load model
        vl_backbone = setup_model(cfg)
        logging.info("Loaded backbone according to config {}".format(backbone_config_json))
        return vl_backbone

### --------- Patch & Fuse: Singularity --------- ###
import copy
from Singularity.models.tokenization_bert import BertTokenizer
from Singularity.models.utils import interpolate_pos_embed, interpolate_pos_relative_bias_beit, load_temp_embed_with_mismatch
from Singularity.models.model_retrieval import Singularity

@registry.register_model("patch_and_fuse_singularity")
class PaxionSingularity(PaxionInternVideo):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "singularity_temporal_17m": "configs/models/patch_and_fuse_singularity.yaml",
    }
    def __init__(
        self,
        vl_backbone,
        **kwargs
    ):
        super().__init__(vl_backbone, **kwargs)
    
    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return self.vl_backbone.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            return_attention_mask=True
        )

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        text_embeds, pooled_text_embeds = self.vl_backbone.encode_text(text)
        # pooled_text_feats = F.normalize(self.vl_backbone.text_proj(pooled_text_embeds), dim=-1) # B, d
        pooled_text_feats = self.vl_backbone.text_proj(pooled_text_embeds)
        if return_all_feats:
            return pooled_text_feats, text_embeds
        return pooled_text_feats
      
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        B, num_frm, C, H, W = video.shape
        video_embeds, pooled_video_embeds = self.vl_backbone.encode_image(video)
        video_embeds = rearrange(video_embeds, "b (f p) d -> p b f d", b=B, f=num_frm)
        # pooled_video_feats = F.normalize(self.vl_backbone.vision_proj(pooled_video_embeds), dim=-1) # B, #frm, d
        pooled_video_feats = self.vl_backbone.vision_proj(pooled_video_embeds) # B, #frm, d
        pooled_video_feats = pooled_video_feats.mean(dim=1) # B, d # mean pooling for consistency with other backbone
        if return_all_feats:
            return pooled_video_feats, video_embeds
        return pooled_video_feats

    @classmethod
    def _load_backbone(cls, cfg):
        def setup_model(config, model_cls, has_decoder=False):
            print("Creating model")
            config = copy.deepcopy(config)

            tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
            model_without_ddp = model_cls(config=config, tokenizer=tokenizer)

            assert config.pretrained_path is not None
            print(f"Loading checkpoint from {config.pretrained_path}")
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            state_dict = checkpoint["model"]

            # reshape positional embeddings
            is_beit = "beit" in config.vit_type
            if is_beit:
                # interpolate relative pos bias
                state_dict = interpolate_pos_relative_bias_beit(
                    state_dict_old=state_dict,
                    state_dict_new=model_without_ddp.state_dict(),
                    patch_shape_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.patch_shape
                )
            else:
                # interpolate pos_embed
                state_dict["vision_encoder.embeddings.position_embeddings"] = \
                    interpolate_pos_embed(
                        pos_embed_old=state_dict["vision_encoder.embeddings.position_embeddings"],
                        pos_embed_new=model_without_ddp.vision_encoder.embeddings.position_embeddings,
                        num_patches_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.num_patches
                    )

            for key in list(state_dict.keys()):
                if "bert" in key:
                    encoder_key = key.replace("bert.", "")
                    state_dict[encoder_key] = state_dict[key]
                    if not has_decoder:
                        del state_dict[key]

                # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                # only for generation tasks like VQA
                if has_decoder and "text_encoder" in key:
                    if "layer" in key:
                        encoder_keys = key.split(".")
                        layer_num = int(encoder_keys[4])
                        if layer_num < 9:  # configs/config_bert.fusion_layer
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num-9)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = ".".join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

            # load temporal_embeddings, clip or expand when necessary
            state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
                temp_embed_old=state_dict["temporal_embeddings"],
                temp_embed_new=model_without_ddp.temporal_embeddings.data
            )

            # print("all loaded keys:")
            # for key in list(state_dict.keys()):
            #     print("\t", key)

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            print("missing_keys:", msg.missing_keys)
            print("unexpected_keys:", msg.unexpected_keys)
            print(f"Loaded checkpoint from {config.pretrained_path}")
            
            return model_without_ddp, tokenizer

        backbone_config_yaml = cfg.get("backbone_config_yaml", None)
        assert backbone_config_yaml != None
        cfg = OmegaConf.load(backbone_config_yaml)
        print("model cfg:", cfg)
        vl_backbone, tokenizer = setup_model(cfg, model_cls=Singularity)
        return vl_backbone



### --------- KnowledgePatcherInternVideo Baseline: Transformer Patcher --------- ###
class TransformerPatcher(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True # expect batch_size as first dimention of the input
            ),
            num_layers=num_layers,
        )
    def forward(self, x):
        output = self.transformer_encoder(x)
        return output

@registry.register_model("patch_and_fuse_internvideo_baseline")
class KnowledgePatcherInternVideo_Baseline(KnowledgePatcherBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "InternVideo-MM-L-14": "configs/models/patch_and_fuse_intern_video.yaml",
    }
    def __init__(self, 
        vl_backbone
    ):
        super().__init__(
            vl_backbone,
        )
        self.vision_adaptor = TransformerPatcher(input_dim=1024, hidden_dim=2048, num_heads=8, num_layers=1)
        self.text_adaptor = TransformerPatcher(input_dim=768, hidden_dim=2048, num_heads=8, num_layers=1)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
    
    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return InternVideo.tokenize(text, truncate=True, return_special_tokens_mask=False)
    
    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        return self.vl_backbone.encode_text(text, return_all_feats=return_all_feats, masked_indices=masked_indices)
    
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        # set num_frm in backbone visiontransformer
        num_frm = video.shape[1]
        self.vl_backbone.visual.transformer.T = num_frm
        
        video = rearrange(video, 'b m c h w -> b c m h w') # InternVideo expect input to be (B, C, num_frm, H, W)
        return self.vl_backbone.encode_video(video, return_all_feats=return_all_feats, mode=mode)

    def encode_text(self, text, return_all_feats=True, masked_indices=None, **kwargs):
        raw_text_feat, raw_text_embeddings = self.encode_text_backbone(
            text, 
            return_all_feats=True, 
            masked_indices=masked_indices
        )
        adapted_text_embeddings = self.text_adaptor(raw_text_embeddings)
        # print("adapted_text_embeddings.shape:", adapted_text_embeddings.shape)
        adapted_text_feats = adapted_text_embeddings[torch.arange(adapted_text_embeddings.shape[0]), text.argmax(dim=-1)]
        # print("adapted_text_feats.shape:", adapted_text_feats.shape)
        if self.vl_backbone.text_projection is not None:
            adapted_text_feats = adapted_text_feats @ self.vl_backbone.text_projection
        if return_all_feats:
            return adapted_text_feats, adapted_text_embeddings
        return adapted_text_feats
            
    def encode_video(self, video, return_all_feats=True, masked_indices=None, **kwargs):
        # set num_frm in backbone visiontransformer
        num_frm = video.shape[1]
        self.vl_backbone.visual.transformer.T = num_frm
        
        video = rearrange(video, 'b m c h w -> b c m h w') # InternVideo expect input to be (B, C, num_frm, H, W)
        raw_visual_feature, raw_visual_embeddings = self.vl_backbone.visual(video, return_all_feats=True, mode="video")
        # go through adaption layer
        raw_visual_embeddings = rearrange(raw_visual_embeddings, "p b f d -> b (f p) d")
        adapted_visual_embeddings = self.vision_adaptor(raw_visual_embeddings) # B (num_frm*num_patch) dim_v
        # print("adapted_visual_embeddings.shape:", adapted_visual_embeddings.shape)
        adapted_visual_feats = adapted_visual_embeddings.mean(dim=1) # B, dim_v
        # print("adapted_visual_feats.shape:", adapted_visual_feats.shape)
        if self.vl_backbone.visual_proj is not None:
            adapted_visual_feats = adapted_visual_feats @ self.vl_backbone.visual_proj
        if return_all_feats:
            return adapted_visual_feats, adapted_visual_embeddings # (B, dim_v_projected)  (B (num_frm*num_patch) dim_v)
        return adapted_visual_feats

    def forward(self, samples):
        video_input = samples["video_input"]
        text_input = samples["text_input"]

        # backbone features
        text_input_tokenized = self.tokenize(text_input).to(video_input.device)
        
        visual_feat, _ = self.encode_video(video_input)
        text_feat, _ = self.encode_text(text_input_tokenized)
        
        text_feat = F.normalize(text_feat, dim=-1)
        visual_feat = F.normalize(visual_feat, dim=-1)
        
        ###============== Video-text Contrastive ===================###
        visual_feat_all = concat_all_gather(visual_feat)  # [batch_size*num_gpu, emb_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, emb_dim]

        # video-text similarity: aggregate across all query tokens
        sim_v2t = visual_feat @ text_feat_all.T
        sim_v2t = sim_v2t / self.temp # (batch_size, batch_size*num_gpu)
        
        # text-video similarity: aggregate across all query tokens
        sim_t2v = text_feat @ visual_feat_all.T
        sim_t2v = sim_t2v / self.temp  # [batch_size, batch_size*num_gpu]

        rank = get_rank()
        bs = video_input.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            video_input.device
        )

        loss_vtc = (
            F.cross_entropy(sim_v2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2v, targets, label_smoothing=0.1)
        ) / 2

        total_loss = loss_vtc
        losses = {VTC:loss_vtc}

        return total_loss, losses

    @classmethod
    def _load_backbone(cls, cfg):
        logging.info("Model config: {}".format(cfg))
        backbone_pretrained_ckpt = cfg.get("backbone_pretrained", None)
        assert backbone_pretrained_ckpt != None
        vl_backbone = InternVideo.load_model(backbone_pretrained_ckpt)
        logging.info("Loaded backbone pretrained weights from {}".format(backbone_pretrained_ckpt))
        return vl_backbone

    @classmethod
    def from_config(cls, cfg=None):
        vl_backbone = cls._load_backbone(cfg)
        model = cls(
            vl_backbone,
        )
        # load trained ckpt if specified
        pretrained_ckpt = cfg.get("pretrained", None)
        load_pretrained = cfg.get("load_pretrained", True)
        if (pretrained_ckpt is not None) and load_pretrained:
            logging.info("Loading pretrained weights...")
            msg = load_from_pretrained(model, url_or_filename=pretrained_ckpt)
        else:
            logging.info("No additional pretrained weights are loaded.")

        return model

### --------- KnowledgePatcherClipVip Baseline: vision Transformer Patcher --------- ###
@registry.register_model("patch_and_fuse_clipvip_baseline_simple")
class KnowledgePatcherClipVip_Baseline_Simple(KnowledgePatcherInternVideo_Baseline):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vip_base_32": "configs/models/patch_and_fuse_clip_vip.yaml",
    }
    def __init__(self,
        vl_backbone
    ):
        super().__init__(
            vl_backbone,
        )
        self.vision_adaptor = TransformerPatcher(input_dim=768, hidden_dim=1024, num_heads=8, num_layers=1)
        self.text_adaptor = None
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return self.tokenizer(text, padding=True, return_tensors="pt")

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        clipmodel = self.vl_backbone.clipmodel
        text_outputs = clipmodel.text_model(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask'],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        pooled_output = text_outputs.pooler_output # (B, dim_t)
        text_embeddings = text_outputs.last_hidden_state # (B, L, dim_t)
        text_features = clipmodel.text_projection(pooled_output) 
        # noramize
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if return_all_feats:
            return text_features, text_embeddings
        return text_features
    
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        #TODO: debug this
        B, num_frm, C, H, W = video.shape
        clipmodel = self.vl_backbone.clipmodel
        vision_outputs = clipmodel.vision_model(
            pixel_values=video,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        pooled_output = vision_outputs.pooler_output  # (B, dim_v)
        video_embeddings = vision_outputs.last_hidden_state[:,4:,:] # (B, num_frm*P, dim_v); the first 4 elements are the cls tokens
        video_embeddings = rearrange(video_embeddings, 'b (n p) d -> p b n d', n=num_frm)
        video_features = clipmodel.visual_projection(pooled_output)
        # noramize
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        if return_all_feats:
            return video_features, video_embeddings # TODO: check output dim
        return video_features

    def encode_text(self, text, return_all_feats=True, masked_indices=None, **kwargs):
        return self.encode_text_backbone(
            text,
            return_all_feats=return_all_feats, 
            masked_indices=masked_indices
        )

    def encode_video(self, video, return_all_feats=True, masked_indices=None, **kwargs):
        # set num_frm in backbone visiontransformer
        clipmodel = self.vl_backbone.clipmodel

        video_features, raw_visual_embeddings = self.encode_video_backbone(video, return_all_feats=True)
        # go through adaption layer
        raw_visual_embeddings = rearrange(raw_visual_embeddings, "p b f d -> b (f p) d")
        adapted_visual_embeddings = self.vision_adaptor(raw_visual_embeddings) # B (num_frm*num_patch) dim_v
        adapted_visual_feats = clipmodel.visual_projection(adapted_visual_embeddings.mean(dim=1)) # B, d_proj
        if return_all_feats:
            return adapted_visual_feats, adapted_visual_embeddings # (B, dim_v_projected)  (B (num_frm*num_patch) dim_v)
        return adapted_visual_feats

    @classmethod
    def _load_backbone(cls, cfg):
        """ load pretrained CLIP-Vip backbone """
        
        def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
            """operated in-place, no need to return `model`"""

            if isinstance(loaded_state_dict_or_path, str):
                loaded_state_dict = torch.load(
                    loaded_state_dict_or_path, map_location="cpu")
            else:
                loaded_state_dict = loaded_state_dict_or_path
            model_keys = set([k for k in list(model.state_dict().keys())])
            load_keys = set(loaded_state_dict.keys())

            toload = {}
            mismatched_shape_keys = []
            for k in model_keys:
                if k in load_keys:
                    if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                        mismatched_shape_keys.append(k)
                    else:
                        toload[k] = loaded_state_dict[k]

            print("You can ignore the keys with `num_batches_tracked` or from task heads")
            print("Keys in loaded but not in model:")
            diff_keys = load_keys.difference(model_keys)
            print(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
            print("Keys in model but not in loaded:")
            diff_keys = model_keys.difference(load_keys)
            print(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
            print("Keys in model and loaded, but shape mismatched:")
            print(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
            model.load_state_dict(toload, strict=False)
        
        def setup_model(cfg):
            print("Setup model...")
            
            model = VidCLIP(cfg)

            if cfg.e2e_weights_path:
                print(f"Loading e2e weights from {cfg.e2e_weights_path}")
                load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
            
            if hasattr(cfg, "overload_logit_scale"):
                model.overload_logit_scale(cfg.overload_logit_scale)
            
            print("Setup model done!")
            return model

        backbone_config_json = cfg.get("backbone_config_json", None)
        assert backbone_config_json != None
        # parsed_args = shared_configs.get_pretraining_args()
        # cfg = load_config_json_to_args(parsed_args, backbone_config_json)
        # vl_backbone = setup_model(cfg)
        cfg = edict(json.load(open(backbone_config_json)))
        # load model
        vl_backbone = setup_model(cfg)
        logging.info("Loaded backbone according to config {}".format(backbone_config_json))
        return vl_backbone

### --------- KnowledgePatcherSingularity Baseline: vision Transformer Patcher --------- ###
@registry.register_model("patch_and_fuse_singularity_baseline_simple")
class KnowledgePatcherSingularity_Baseline_Simple(KnowledgePatcherInternVideo_Baseline):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "singularity_temporal_17m": "configs/models/patch_and_fuse_singularity.yaml",
    }
    def __init__(self,
        vl_backbone
    ):
        super().__init__(
            vl_backbone,
        )
        self.vision_adaptor = TransformerPatcher(input_dim=768, hidden_dim=1024, num_heads=8, num_layers=1)
        self.text_adaptor = None
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def tokenize(self, text, **kwargs):
        """ text: a list of strings """
        return self.vl_backbone.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            return_attention_mask=True
        )

    def encode_text_backbone(self, text, return_all_feats=False, masked_indices=None):
        """ text: tokenized batch of text 
            return: 
                - if return_all_feats is False: return single text feature vector (B, dim_t)
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_t) (B, L, dim_t) where L is the token length
        """
        text_embeds, pooled_text_embeds = self.vl_backbone.encode_text(text)
        # pooled_text_feats = F.normalize(self.vl_backbone.text_proj(pooled_text_embeds), dim=-1) # B, d
        pooled_text_feats = self.vl_backbone.text_proj(pooled_text_embeds)
        if return_all_feats:
            return pooled_text_feats, text_embeds
        return pooled_text_feats
      
    def encode_video_backbone(self, video, return_all_feats=False, mode="video", masked_indices=None):
        """ video: tensor of size (B, num_frm, C, H, W) 
            return: 
                - if return_all_feats is False: return single video feature vector (B, dim_v) where B is 
                - if return_all_feats is True: return single vector, feature sequence: (B, dim_v) (P, B, num_frm, dim_v) where P is number of patches
        """
        B, num_frm, C, H, W = video.shape
        video_embeds, pooled_video_embeds = self.vl_backbone.encode_image(video)
        video_embeds = rearrange(video_embeds, "b (f p) d -> p b f d", b=B, f=num_frm)
        # pooled_video_feats = F.normalize(self.vl_backbone.vision_proj(pooled_video_embeds), dim=-1) # B, #frm, d
        pooled_video_feats = self.vl_backbone.vision_proj(pooled_video_embeds) # B, #frm, d
        pooled_video_feats = pooled_video_feats.mean(dim=1) # B, d # mean pooling for consistency with other backbone
        if return_all_feats:
            return pooled_video_feats, video_embeds
        return pooled_video_feats

    def encode_text(self, text, return_all_feats=True, masked_indices=None, **kwargs):
        return self.encode_text_backbone(
            text,
            return_all_feats=return_all_feats, 
            masked_indices=masked_indices
        )

    def encode_video(self, video, return_all_feats=True, masked_indices=None, **kwargs):
        # set num_frm in backbone visiontransformer
        video_features, raw_visual_embeddings = self.encode_video_backbone(video, return_all_feats=True)
        # go through adaption layer
        raw_visual_embeddings = rearrange(raw_visual_embeddings, "p b f d -> b (f p) d")
        adapted_visual_embeddings = self.vision_adaptor(raw_visual_embeddings) # B (num_frm*num_patch) dim_v
        adapted_visual_feats = self.vl_backbone.vision_proj(adapted_visual_embeddings.mean(dim=1)) # B, d_proj
        if return_all_feats:
            return adapted_visual_feats, adapted_visual_embeddings # (B, dim_v_projected)  (B (num_frm*num_patch) dim_v)
        return adapted_visual_feats

    @classmethod
    def _load_backbone(cls, cfg):
        def setup_model(config, model_cls, has_decoder=False):
            print("Creating model")
            config = copy.deepcopy(config)

            tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
            model_without_ddp = model_cls(config=config, tokenizer=tokenizer)

            assert config.pretrained_path is not None
            print(f"Loading checkpoint from {config.pretrained_path}")
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            state_dict = checkpoint["model"]

            # reshape positional embeddings
            is_beit = "beit" in config.vit_type
            if is_beit:
                # interpolate relative pos bias
                state_dict = interpolate_pos_relative_bias_beit(
                    state_dict_old=state_dict,
                    state_dict_new=model_without_ddp.state_dict(),
                    patch_shape_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.patch_shape
                )
            else:
                # interpolate pos_embed
                state_dict["vision_encoder.embeddings.position_embeddings"] = \
                    interpolate_pos_embed(
                        pos_embed_old=state_dict["vision_encoder.embeddings.position_embeddings"],
                        pos_embed_new=model_without_ddp.vision_encoder.embeddings.position_embeddings,
                        num_patches_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.num_patches
                    )

            for key in list(state_dict.keys()):
                if "bert" in key:
                    encoder_key = key.replace("bert.", "")
                    state_dict[encoder_key] = state_dict[key]
                    if not has_decoder:
                        del state_dict[key]

                # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                # only for generation tasks like VQA
                if has_decoder and "text_encoder" in key:
                    if "layer" in key:
                        encoder_keys = key.split(".")
                        layer_num = int(encoder_keys[4])
                        if layer_num < 9:  # configs/config_bert.fusion_layer
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num-9)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = ".".join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

            # load temporal_embeddings, clip or expand when necessary
            state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
                temp_embed_old=state_dict["temporal_embeddings"],
                temp_embed_new=model_without_ddp.temporal_embeddings.data
            )

            # print("all loaded keys:")
            # for key in list(state_dict.keys()):
            #     print("\t", key)

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            print("missing_keys:", msg.missing_keys)
            print("unexpected_keys:", msg.unexpected_keys)
            print(f"Loaded checkpoint from {config.pretrained_path}")
            
            return model_without_ddp, tokenizer

        backbone_config_yaml = cfg.get("backbone_config_yaml", None)
        assert backbone_config_yaml != None
        cfg = OmegaConf.load(backbone_config_yaml)
        print("model cfg:", cfg)
        vl_backbone, tokenizer = setup_model(cfg, model_cls=Singularity)
        return vl_backbone

### --------- patch_and_fuse_internvideo_baseline_simple: vision Transformer Patcher --------- ###
@registry.register_model("patch_and_fuse_internvideo_baseline_simple")
class KnowledgePatcherInternVideo_Baseline_Simple(KnowledgePatcherInternVideo_Baseline):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "InternVideo-MM-L-14": "configs/models/patch_and_fuse_intern_video.yaml",
    }
    def __init__(self, 
        vl_backbone
    ):
        super().__init__(
            vl_backbone,
        )
        self.vision_adaptor = TransformerPatcher(input_dim=1024, hidden_dim=2048, num_heads=8, num_layers=1)
        self.text_adaptor = None
        self.temp = nn.Parameter(0.07 * torch.ones([]))
    
    def encode_text(self, text, return_all_feats=True, masked_indices=None, **kwargs):
        return self.encode_text_backbone(
            text,
            return_all_feats=return_all_feats, 
            masked_indices=masked_indices
        )




### --------- Transformer baseline for MCQA --------- ###

def _mcqa_forward_knowledge_patcher_baseline(model, samples):
    video_input = samples["video_input"] # B, num_frm, 3, 224, 224
    text_input = samples["text_input"] # list of list of string (B, 5)
    targets = samples["answer"]

    # get visual features
    video_feat = model.encode_video(video_input, return_all_feats=False)
    video_feat = F.normalize(video_feat, dim=-1)

    # get text features
    text_tensor_batch = [model.tokenize(choices).to(video_feat.device) for choices in text_input]
    text_feats_batch = []
    for text_tensor in text_tensor_batch:
        raw_text_feat, _ = model.encode_text(text_tensor)
        raw_text_feat = F.normalize(raw_text_feat, dim=-1)
        text_feats_batch.append(raw_text_feat)
    
    sims = []
    for i in range(len(text_tensor_batch)):
        v_feat = video_feat[i].unsqueeze(0) # (1, D) or (1, num_latents, D)
        t_feat = text_feats_batch[i] # (5, D) or (5, num_latents, D)
        sim_v2t_b = v_feat @ t_feat.T / model.temp
        assert sim_v2t_b.shape == (1,5)
        sims.append(sim_v2t_b)
    sims = torch.cat(sims, dim=0)

    mcqa_loss = F.cross_entropy(sims, targets.to(sims.device))
    losses = {"mcqa_loss":mcqa_loss}
    return mcqa_loss, losses

@registry.register_model("patch_and_fuse_internvideo_baseline_simple_mcqa")
class KnowledgePatcherInternVideo_Baseline_Simple_MCQA(KnowledgePatcherInternVideo_Baseline):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "InternVideo-MM-L-14": "configs/models/patch_and_fuse_intern_video.yaml",
    }
    def __init__(self, 
        vl_backbone
    ):
        super().__init__(
            vl_backbone,
        )
        self.vision_adaptor = TransformerPatcher(input_dim=1024, hidden_dim=2048, num_heads=8, num_layers=1)
        self.text_adaptor = None
        self.temp = nn.Parameter(0.07 * torch.ones([]))
    
    def encode_text(self, text, return_all_feats=True, masked_indices=None, **kwargs):
        return self.encode_text_backbone(
            text,
            return_all_feats=return_all_feats, 
            masked_indices=masked_indices
        )

    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher_baseline(self, samples)

@registry.register_model("patch_and_fuse_clipvip_baseline_simple_mcqa")
class KnowledgePatcherClipVip_Baseline_Simple_MCQA(KnowledgePatcherClipVip_Baseline_Simple):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vip_base_32": "configs/models/patch_and_fuse_clip_vip.yaml",
    }
    def __init__(self, vl_backbone):
        super().__init__(vl_backbone)
    
    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher_baseline(self, samples)

@registry.register_model("patch_and_fuse_singularity_baseline_simple_mcqa")
class KnowledgePatcherSingularity_Baseline_Simple_MCQA(KnowledgePatcherSingularity_Baseline_Simple):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "singularity_temporal_17m": "configs/models/patch_and_fuse_singularity.yaml",
    }
    def __init__(self, vl_backbone):
        super().__init__(vl_backbone)
    
    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher_baseline(self, samples)


### --------- Patch and Fuse for MCQA --------- ###

def _mcqa_get_answer_logits_knowledge_patcher(model, visual_feats, text_features):
        # visual_feat: (1, D) or (1, Q, D)
        # text_feat: (5, D) or (5, Q, D)
        if text_features.dim() == 2:
            if visual_feats.dim() == 2:
                sim_v2t = (visual_feats @ text_features.T) / model.temp
            elif visual_feats.dim() == 3:
                sim_q2t = torch.matmul(
                    visual_feats.unsqueeze(1), # (M, 1, num_query_tokens, latent_dim)  
                    text_features.unsqueeze(-1) # (N, latent_dim, 1)
                ).squeeze(-1) # (M, N, num_query_tokens)
                sim_v2t, _ = sim_q2t.max(-1)
                sim_v2t = sim_v2t / model.temp # (M, N)
        else:
            if visual_feats.dim() == 2:
                sim_v2q = torch.matmul(
                    visual_feats.unsqueeze(1).unsqueeze(1), # (M, 1, 1, latent_dim)  
                    rearrange(text_features, "c l d -> c d l") # (N, latent_dim, num_latents)
                ).squeeze(-2) # (M, N, num_latents)
                sim_v2t, _ = sim_v2q.max(-1) # (M, N)
                sim_v2t = sim_v2t / model.temp # (M, N)
            elif visual_feats.dim() == 3:
                sim_v2q = torch.matmul(
                    visual_feats.unsqueeze(1), # (M, 1, num_latents, latent_dim)  
                    rearrange(text_features, "c l d -> c d l") # (N, latent_dim, num_latents)
                ) # (M, N, num_latents, num_latents)
                sim_v2t, _ = sim_v2q.mean(dim=-1).max(dim=-1)
                sim_v2t = sim_v2t / model.temp # (M, N)
        assert text_features.shape[0] > 1
        # probs = sim_v2t.softmax(dim=-1) # (M, N)
        return sim_v2t # (1, 5) | (1, 10) if with antonym

def _mcqa_forward_knowledge_patcher(model, samples):
    video_input = samples["video_input"] # (B, num_frm, C, H, W) 
    text_input = samples["text_input"]
    targets = samples["answer"]

    # get text features
    text_tensor_batch = [model.tokenize(choices).to(video_input.device) for choices in text_input]
    text_feats_batch = []
    for text_tensor in text_tensor_batch:
        raw_text_feat, perceiver_t_feat = model.encode_text(text_tensor)
        raw_text_feat = F.normalize(raw_text_feat, dim=-1)
        perceiver_t_feat = F.normalize(perceiver_t_feat, dim=-1)
        if model.if_use_dual_perceiver:
            text_feats_batch.append(perceiver_t_feat)
        else:
            text_feats_batch.append(raw_text_feat)

    ## get video features
    raw_visual_feat, percevier_visual_embeddings = model.encode_video(video_input)
    video_feat = F.normalize(percevier_visual_embeddings, dim=-1)

    losses = {}
    assert "mcqa_loss" in model.loss_weighting
    if VAC in model.loss_weighting and not model.if_as_knowledge_fuser:
        # pad the action_antonym_text_input to length of five 
        # by copying over candidate answers that are not ground truth 
        action_antonym_text_input = samples["action_antonym_text_input"]
        for i in range(len(action_antonym_text_input)):
            ans = targets[i].item()
            incorrect_cands = [text_input[i][j] for j in range(5) if j != ans]
            if len(action_antonym_text_input[i]) < 5:
                incorrect_cands_repeat = incorrect_cands + incorrect_cands # len = 8
                action_antonym_text_input[i] += incorrect_cands_repeat
                action_antonym_text_input[i] = action_antonym_text_input[i][:5]
            assert len(action_antonym_text_input[i]) == 5
        # get text antonym features
        action_antonym_text_tensor_batch = [model.tokenize(choices).to(video_input.device) for choices in action_antonym_text_input]
        antonym_text_feats_batch = []
        for text_tensor in action_antonym_text_tensor_batch:
            antonym_raw_text_feat, antonym_perceiver_t_feat = model.encode_text(text_tensor)
            antonym_raw_text_feat = F.normalize(antonym_raw_text_feat, dim=-1)
            antonym_perceiver_t_feat = F.normalize(antonym_perceiver_t_feat, dim=-1)
            if model.if_use_dual_perceiver:
                antonym_text_feats_batch.append(antonym_perceiver_t_feat)
            else:
                antonym_text_feats_batch.append(antonym_raw_text_feat)
        assert len(antonym_text_feats_batch) == len(text_feats_batch)
        text_feats_batch_final = [
            torch.cat([text_feats_batch[i],antonym_text_feats_batch[i]], dim=0) # (5+5,D) | (5+5,Q,D)
            for i in range(len(text_feats_batch))
        ]
    else:
        text_feats_batch_final = text_feats_batch
    
    ## get logits
    sims = []
    for i in range(len(text_feats_batch_final)):
        v_feat = video_feat[i].unsqueeze(0) # (1, D) or (1, num_latents, D)
        t_feat = text_feats_batch_final[i] # (5, D) or (5, num_latents, D) | (10, D) or (10, num_latents, D) if using inverse dm
        sim_v2t_b = _mcqa_get_answer_logits_knowledge_patcher(model, v_feat, t_feat) # (1,5) | (1,10)
        sims.append(sim_v2t_b)
    sims = torch.cat(sims, dim=0)
    assert sims.shape[0] == targets.shape[0]
    
    mcqa_loss = F.cross_entropy(sims, targets.to(sims.device), label_smoothing=0.1)
    losses = {"mcqa_loss":mcqa_loss}
    return mcqa_loss, losses

@registry.register_model("patch_and_fuse_internvideo_mcqa")
class PaxionInternVideo_MCQA(PaxionInternVideo):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "InternVideo-MM-L-14": "configs/models/patch_and_fuse_intern_video.yaml",
    }
 
    def __init__(self, 
        vl_backbone,
        **kwargs
    ):
        super().__init__(
            vl_backbone,
            **kwargs
        )

    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher(self, samples)

@registry.register_model("patch_and_fuse_clipvip_mcqa")
class PaxionClipVIP_MCQA(PaxionClipVIP):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vip_base_32": "configs/models/patch_and_fuse_clip_vip.yaml",
    }

    def __init__(self, 
        vl_backbone,
        **kwargs
    ):
        super().__init__(
            vl_backbone,
            **kwargs
        )

    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher(self, samples)

@registry.register_model("patch_and_fuse_singularity_mcqa")
class PaxionSingularity_MCQA(PaxionSingularity):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "singularity_temporal_17m": "configs/models/patch_and_fuse_singularity.yaml",
    }

    def __init__(self, 
        vl_backbone,
        **kwargs
    ):
        super().__init__(
            vl_backbone,
            **kwargs
        )

    def forward(self, samples):
        return _mcqa_forward_knowledge_patcher(self, samples)
