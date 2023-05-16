from models import * 
from processors import * 
from lavis.models import load_model_and_preprocess, load_preprocess
from transformers import set_seed
from data import *
from lavis.common.config import Config
from lavis.common.registry import registry

def set_up_device(gpu_index):
    # single gpu
    if torch.cuda.is_available() and gpu_index != -1:
        dev = f"cuda:{gpu_index}"
    else:
        dev = "cpu"
    return torch.device(dev)

def load_preprocess_from_register_name(name, model_type, device="cpu"):
    """
    Args:
        name (str): name of the model.
        model_type (str): type of the model.
    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )
    return vis_processors, txt_processors

def compute_sim_main_patcher_and_fuser(model, text_features, fused_visual_features, v2t=True):
    # assert fused_visual_features.dim() == 2
    if v2t:
        if text_features.dim() == 2:
            if fused_visual_features.dim() == 2:
                sim_v2t = (fused_visual_features @ text_features.T) / model.temp
            elif fused_visual_features.dim() == 3:
                sim_q2t = torch.matmul(
                    fused_visual_features.unsqueeze(1), # (M, 1, num_query_tokens, latent_dim)  
                    text_features.unsqueeze(-1) # (N, latent_dim, 1)
                ).squeeze(-1)
                sim_v2t, _ = sim_q2t.max(-1)
                sim_v2t = sim_v2t / model.temp # (M, N)
        else:
            if fused_visual_features.dim() == 2:
                sim_v2q = torch.matmul(
                    fused_visual_features.unsqueeze(1).unsqueeze(1), # (M, 1, 1, latent_dim)  
                    rearrange(text_features, "c l d -> c d l") # (N, latent_dim, num_latents)
                ).squeeze(-2) # (M, N, num_latents)
                sim_v2t, _ = sim_v2q.max(-1) # (M, N)
                sim_v2t = sim_v2t / model.temp # (M, N)
            elif fused_visual_features.dim() == 3:
                sim_v2q = torch.matmul(
                    fused_visual_features.unsqueeze(1), # (M, 1, num_latents, latent_dim)  
                    rearrange(text_features, "c l d -> c d l") # (N, latent_dim, num_latents)
                ) # (M, N, num_latents, num_latents)
                sim_v2t, _ = sim_v2q.mean(dim=-1).max(dim=-1)
                sim_v2t = sim_v2t / model.temp # (M, N)
        assert text_features.shape[0] > 1
        probs = sim_v2t.softmax(dim=-1).detach().cpu().numpy() # (M, N)

    else:
        if text_features.dim() == 2:
            if fused_visual_features.dim() == 2:
                sim_t2v = (text_features @ fused_visual_features.T) / model.temp # (N, M)
            elif fused_visual_features.dim() == 3:
                sim_t2q = torch.matmul(
                    text_features.unsqueeze(1).unsqueeze(1), # (N, 1, 1, latent_dim,)
                    rearrange(fused_visual_features, "c q d -> c d q")  # (M, latent_dim, num_query_tokens)
                ).squeeze(-2)
                sim_t2v, _ = sim_t2q.max(-1)
                sim_t2v = sim_t2v / model.temp  # (N, M)
        else:
            if fused_visual_features.dim() == 2:
                sim_q2v = torch.matmul(
                    text_features.unsqueeze(1), # (N, 1, num_latents, latent_dim)  
                    fused_visual_features.unsqueeze(-1) # (M, latent_dim, 1)
                ).squeeze(-1) # (N, M, num_latents)
                sim_t2v, _ = sim_q2v.max(dim=-1) # (N, M)
                sim_t2v = sim_t2v / model.temp # (N, M)
            elif fused_visual_features.dim() == 3:
                sim_q2v = torch.matmul(
                    text_features.unsqueeze(1), # (N, 1, num_latents, latent_dim)  
                    rearrange(fused_visual_features, "c l d -> c d l") # (M, latent_dim, num_latents)
                ).squeeze(-1) # (N, M, num_latents, num_latents)
                sim_t2v, _ = sim_q2v.mean(dim=-1).max(dim=-1) # (N, M)
                sim_t2v = sim_t2v / model.temp # (N, M)
        assert fused_visual_features.shape[0] > 1
        probs = sim_t2v.softmax(dim=-1).detach().cpu().numpy() # (N, M)
    return probs


if __name__ == "__main__":
    decord.bridge.set_bridge('native')
    
    # set up device
    device = set_up_device(gpu_index=0)

    # input video
    video_path = "demo_video/ssv2_194058__book_falling_like_a_rock.mp4"
    
    # input text
    text_input = ["Book falling like a rock", "Book rising like a rock"]
    
    # set up config
    # NOTE: make sure to set the "pretrained" field to the path of the downloaded checkpoint: "pretrained_ckpt/PatchAndFuse/downstream_tasks/ssv2_label_patch_and_fuse.pth"
    config_path = "configs/projects/eval/downstream_task/ssv2_label/patch_and_fuse.yaml"
    config = OmegaConf.load(config_path)
    model_config = Config.build_model_config(config)["model"]
    model_name = model_config.arch
    model_type = model_config.model_type

    ## == load vision and text processor == ##
    vis_processors, txt_processors = load_preprocess_from_register_name(
        name=model_name, 
        model_type=model_type
    )
    vis_processor = vis_processors['eval']
    txt_processor = txt_processors['eval']
    print("vis_processor class:", vis_processor)
    print("txt_processor class:", txt_processor)

    ## == load model from config == ##
    model_cls = registry.get_model_class(model_name)
    print("model class:", model_cls)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    ## == process video and text == ##
    vr = VideoReader(video_path, width=224, height=224)
    vlen = len(vr)
    frame_indices = np.linspace(1,vlen-1,num=8,dtype=int)
    raw_sample_frms = vr.get_batch(frame_indices).asnumpy()
    raw_sample_frms = [Image.fromarray(item, mode="RGB") for item in raw_sample_frms] # PIL Images
    
    video_input = vis_processor(raw_sample_frms).unsqueeze(0).to(device) # tensor: (1, 8, 3, 224, 224)
    text_input = [txt_processor(item) for item in text_input] # list of str
    print("video_input.shape:", video_input.shape)
    print("text_input:", text_input)

    ## == inference == ##
    
    # text features
    text_cand_tokenized = model.tokenize(text_input).to(device)
    text_feats, _ = model.encode_text(text_cand_tokenized)
    text_feats = F.normalize(text_feats, dim=-1)
    print("text_feats.shape:", text_feats.shape)
    
    # visual features
    backbone_video_feat, video_feat = model.encode_video(video_input)
    video_feat = F.normalize(video_feat, dim=-1)
    print("video_feat.shape:", video_feat.shape)

    # compute similarity
    sim_v2t = compute_sim_main_patcher_and_fuser(model, text_feats, video_feat, v2t=True)[0] # (num_text_cand, )

    # print the similarity for each text candidates
    for i, text in enumerate(text_input):
        print("text candidate:", text, "| score:", sim_v2t[i])
