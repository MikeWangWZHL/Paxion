import json
import os
from data import *
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

from einops import rearrange, repeat
from tqdm import tqdm

def compute_sim_start_end_state(model, video_input, text_input):
    """
        video_input: tensor (B, num_frm, C, H, W)
        text_input: list of strings
    """
    bs, num_frm = video_input.shape[0], video_input.shape[1]
    video_input = rearrange(video_input, "b f c h w -> (b f) c h w").to(model.device)
    sample = {"image": video_input, "text_input": text_input}

    features_frames = model.extract_features(sample, mode="image")
    features_text = model.extract_features(sample, mode="text")
    
    features_frames_proj = features_frames.image_embeds_proj[:,0,:] # (B * num_frm, 256)
    features_frames_proj = rearrange(features_frames_proj, "(b f) d -> b f d", b = bs, f=num_frm)
    features_text_proj =  features_text.text_embeds_proj[:,0,:] # (B, 256)

    start_state_frames_feats = features_frames_proj[:,:int(num_frm)//2,:] # (B, num_frm//2, 256)
    end_state_frames_feats = features_frames_proj[:,int(num_frm)//2:,:] # (B, num_frm//2, 256)

    ### compute v-t alignment between frames and text
    start_state_2_t = torch.matmul(
        start_state_frames_feats, # (B, num_frm//2, 256)
        features_text_proj.unsqueeze(-1) # (B, 256, 1)
    ).squeeze(-1) # (B, num_frm, 1)
    start_state_2_t = start_state_2_t.mean(-1) # (B,)
    
    end_state_2_t = torch.matmul(
        end_state_frames_feats, # (B, num_frm//2, 256)
        features_text_proj.unsqueeze(-1) # (B, 256, 1)
    ).squeeze(-1) # (B, num_frm, 1)
    end_state_2_t = end_state_2_t.mean(-1) # (B,)

    ### compute v-v alignment between start and end state
    start_2_end = torch.matmul(
        start_state_frames_feats, # (B, num_frm//2, 256)
        rearrange(end_state_frames_feats, "b f d -> b d f") # (B, 256, num_frm//2)
    ) # (B, num_frm//2, num_frm//2)
    start_2_end = start_2_end.mean(-1).mean(-1) # (B,)

    v_2_v_sim = start_2_end 

    # we wnat v_2_t_sim to be high and v_2_v_sim to be low
    return (
        start_state_2_t.detach().cpu().numpy(),
        end_state_2_t.detach().cpu().numpy(),
        v_2_v_sim.detach().cpu().numpy()
    )

def _reverse_normalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]):
    # reverse the normalization in vis_processor for visualization of the sampled frames
    reverse_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return reverse_normalize(tensor)

def get_concat_h(imgs):
    dst = Image.new('RGB', (sum([imgs[i].width for i in range(len(imgs))]), imgs[0].height))
    prev_img_end_idx = 0
    for i, img in enumerate(imgs):
        dst.paste(img, (prev_img_end_idx, 0))
        prev_img_end_idx += img.width
    return dst


class FilterMetric:
    def __init__(
        self, 
        state_2_text_sim_diff_threshold,
        state_2_state_sim_threshold
    ):
        self.state_2_text_sim_diff_threshold = state_2_text_sim_diff_threshold
        self.state_2_state_sim_threshold = state_2_state_sim_threshold
    
    def if_keep(
        self,
        state_2_t_sim_diff,
        state_2_state_sim
    ):
        if state_2_t_sim_diff > self.state_2_text_sim_diff_threshold and state_2_state_sim < self.state_2_state_sim_threshold:
            return True
        else:
            return False

def _main_loop_visualization(model, metric, dataloader, bs, output_dir, if_output_vis=True, ):

    os.makedirs(output_dir, exist_ok=True)

    annotations = defaultdict(dict)
    keep_vs_discard = [0, 0]
    for i, sample in enumerate(dataloader):
        print(i)
        print(sample['text_input'])
        print(sample['video_input'].shape)
        print()
        
        # compute sims
        start_state_2_t, end_state_2_t, v_2_v_sim = compute_sim_start_end_state(model, sample['video_input'], sample['text_input'])

        print("v_2_v_sim:", v_2_v_sim)

        state_2_t_sim_diff = np.absolute(start_state_2_t - end_state_2_t)
        print("state_2_t_sim_diff:", state_2_t_sim_diff)

        if if_output_vis:
            for b in range(bs):
                video_frames = sample['video_input'][b]

                imgs = [transforms.ToPILImage()(_reverse_normalize(video_frames[i])) for i in range(len(video_frames))]
                get_concat_h(imgs).save(f"{output_dir}/{i}_{b}.jpg")

                if_keep = metric.if_keep(float(state_2_t_sim_diff[b]), float(v_2_v_sim[b]))
                if if_keep:
                    keep_vs_discard[0] += 1
                else:
                    keep_vs_discard[1] += 1

                annotations[f"{i}_{b}"] = {
                    "text_input": sample['text_input'][b],
                    "start_state_2_t": float(start_state_2_t[b]),
                    "end_state_2_t": float(end_state_2_t[b]),
                    "state_2_t_sim_diff": float(state_2_t_sim_diff[b]),
                    # "v_2_t_sim_mean": float(v_2_t_sim_mean[b]),
                    # "v_2_t_sim_max": float(v_2_t_sim_max[b]),
                    "v_2_v_sim": float(v_2_v_sim[b]),
                    "if_keep": if_keep
                }

            with open(f"{output_dir}/annotations.json", 'w') as f:
                json.dump(annotations, f, indent=4)

        if i == 4:
            break
    print("keep_vs_discard:", keep_vs_discard)

def _main_loop_ssv2(model, metric, dataloader, bs, output_dir, split):

    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"state_change_heavy_instance_filtering_{split}.json")

    output_dict = defaultdict(dict)
    keep_vs_discard = [0, 0]

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):        
        # compute sims
        start_state_2_t, end_state_2_t, v_2_v_sim = compute_sim_start_end_state(model, sample['video_input'], sample['text_input'])
        state_2_t_sim_diff = np.absolute(start_state_2_t - end_state_2_t)

        ann_batch = dataloader.dataset.annotation[i*bs : min((i+1)*bs, len(dataloader.dataset)) ]
        
        # print(sample['text_input'])
        # print([item["label"] for item in ann_batch])
        
        for b in range(len(ann_batch)):

            if_keep = metric.if_keep(float(state_2_t_sim_diff[b]), float(v_2_v_sim[b]))

            if if_keep:
                keep_vs_discard[0] += 1
            else:
                keep_vs_discard[1] += 1

            clip_uid = ann_batch[b]['clip_uid']
            output_dict[clip_uid] = {
                "clip_uid": clip_uid,
                "start_state_2_t": float(start_state_2_t[b]),
                "end_state_2_t": float(end_state_2_t[b]),
                "state_2_t_sim_diff": float(state_2_t_sim_diff[b]),
                "v_2_v_sim": float(v_2_v_sim[b]),
                "if_keep": if_keep
            }
        

    print("keep_vs_discard:", keep_vs_discard)
    print("length:", len(output_dict))
    
    with open(output_json_path, 'w') as f:
        json.dump(output_dict, f)

def _main_loop_ego4d(model, metric, dataloader, bs, output_dir, split):

    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"state_change_heavy_instance_filtering_{split}.json")

    output_dict = defaultdict(dict)
    keep_vs_discard = [0, 0]

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):        
        # compute sims
        start_state_2_t, end_state_2_t, v_2_v_sim = compute_sim_start_end_state(model, sample['video_input'], sample['text_input'])
        state_2_t_sim_diff = np.absolute(start_state_2_t - end_state_2_t)

        ann_batch = dataloader.dataset.annotation[i*bs : min((i+1)*bs, len(dataloader.dataset)) ]

        for b in range(len(ann_batch)):

            if_keep = metric.if_keep(float(state_2_t_sim_diff[b]), float(v_2_v_sim[b]))

            if if_keep:
                keep_vs_discard[0] += 1
            else:
                keep_vs_discard[1] += 1

            clip_uid = ann_batch[b]['clip_uid']
            clip_relative_start = ann_batch[b]['clip_relative_start']
            clip_relative_end = ann_batch[b]['clip_relative_end']

            instance_uid = clip_uid + "__" + str(clip_relative_start) + "__" + str(clip_relative_end)
            output_dict[instance_uid] = {
                "clip_uid": clip_uid,
                "clip_relative_start": clip_relative_start,
                "clip_relative_end": clip_relative_end,
                "start_state_2_t": float(start_state_2_t[b]),
                "end_state_2_t": float(end_state_2_t[b]),
                "state_2_t_sim_diff": float(state_2_t_sim_diff[b]),
                "v_2_v_sim": float(v_2_v_sim[b]),
                "if_keep": if_keep
            }

    print("keep_vs_discard:", keep_vs_discard)
    print("length:", len(output_dict))

    with open(output_json_path, 'w') as f:
        json.dump(output_dict, f)

def load_ssv2_dataset(vis_processors, txt_processors, bs,  split = "train"):
    output_vis_dir = "testing_outputs/physical_knowledge_evaulation_dataset/identify_state_change_heavy_videos/ssv2"
    labels_path = '../ActionBench/ssv2/shuffled_object_and_action_antonyms'
    vis_path = '../datasets/SSv2/video_clips/clips_downsampled_5fps_downsized_224x224'
    fps = 5

    if split == "test":
        use_templates_as_labels = True
    else:
        use_templates_as_labels = False

    dataset = ActionBenchDataset_SSv2(
        vis_path, 
        labels_path, 
        vis_processor = vis_processors["eval"],
        text_processor = txt_processors["eval"],
        use_templates_as_labels=use_templates_as_labels, 
        split=split, 
        task="video_text_matching",
        fps=fps
    )
    collator_func = getattr(dataset,"collater",None)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=8,
        collate_fn=collator_func,
        shuffle=False
    )
    return dataloader, output_vis_dir

def load_ego4d_dataset(vis_processors, txt_processors, bs, split = "train"):
    output_vis_dir = "testing_outputs/physical_knowledge_evaulation_dataset/identify_state_change_heavy_videos/ego4d"
    VIS_ROOT = "../datasets/Ego4d/video_clips/clips_downsampled_5fps_downsized_224x224"
    ANN_JSONL = f"../ActionBench/ego4d/egoclip_subset_action_antonyms_train_val_test_split/{split}.jsonl"
    fps = 5

    dataset = ActionBenchDataset_Ego4D(
        vis_root=VIS_ROOT,
        ann_path=ANN_JSONL,
        task = "video_text_matching",
        vis_processor = vis_processors["eval"],
        text_processor = txt_processors["eval"],
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = 224,
        frame_width = 224,
        k=None,
        fps=fps,
        neg_sampling_same_clip=0
    )

    collator_func = getattr(dataset,"collater",None)
    print(collator_func)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=8,
        collate_fn=collator_func,
        shuffle=False
    )
    return dataloader, output_vis_dir


if __name__ == "__main__":

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    ### load blip model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

    ### load dataset
    bs = 16

    ### === visualization === ###
    # # # ssv2
    # dataloader, output_vis_dir = load_ssv2_dataset(vis_processors, txt_processors, bs)
    # metric = FilterMetric(0.003, 0.95)
    
    # print("SSv2...")
    # _main_loop_visualization(model, metric, dataloader, bs, output_vis_dir, if_output_vis=True)
    
    # # ego4d
    # print("Ego4D...")
    # dataloader, output_vis_dir = load_ego4d_dataset(vis_processors, txt_processors, bs)
    # metric = FilterMetric(0.003, 0.95)
    # _main_loop_visualization(model, metric, dataloader, bs, output_vis_dir, if_output_vis=True)
    

    ### === usage example === ###
    # ssv2 train
    split = 'train'
    dataloader, output_vis_dir = load_ssv2_dataset(vis_processors, txt_processors, bs, split)
    output_dir = "../ActionBench/ssv2/shuffled_object_and_action_antonyms"
    metric = FilterMetric(0.003, 0.95)
    _main_loop_ssv2(model, metric, dataloader, bs,  output_dir, split)

    # ego4d train
    split = 'train'
    dataloader, output_vis_dir = load_ego4d_dataset(vis_processors, txt_processors, bs, split)
    output_dir = "../ActionBench/ego4d/egoclip_subset_action_antonyms_train_val_test_split"
    metric = FilterMetric(0.003, 0.95)
    _main_loop_ego4d(model, metric, dataloader, bs, output_dir, split)