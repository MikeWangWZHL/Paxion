"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

# import our modules for registration
from processors import *
from tasks import *
from runners import *
from models import * 
from builders import * 
from data import * 
from tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--inference_type", required=True, help="inference task type")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base_patch_and_fuse"))

    return runner_cls

def inference_step_actionbench(task, model, samples, eval_task, eval_module):
    if eval_task in ["action_antonym", "object_shuffle"]:
        batch_size = len(samples['text_input'])
        if eval_module == "backbone":
            compute_sim_func = task._compute_sim_backbone
        elif eval_module == "knowledge_patcher":                
            if hasattr(model, "if_as_knowledge_fuser") and model.if_as_knowledge_fuser == True:
                compute_sim_func = task._compute_sim_main_fuser
            else:
                compute_sim_func = task._compute_sim_main
        elif eval_module == "knowledge_patcher_baseline":
            compute_sim_func = task._compute_sim_main_baseline

        preds = []
        targets = [0 for b in range(batch_size)]
        
        ### v1 ### using contrastive loss
        for b in range(batch_size):
            video_tensor = samples['video_input'][b].unsqueeze(0)
            if eval_task == 'action_antonym':
                text_cand = [samples["text_input"][b],samples["action_antonym_text_input"][b]]
            else:
                text_cand = [samples["text_input"][b],samples["object_shuffled_text_input"][b]]
            text_tensor = model.tokenize(text_cand).to(video_tensor.device)
            sims = compute_sim_func(model, text_tensor, video_tensor) # (1, 2)
            assert sims.shape == (1,2)
            pred = np.argmax(sims[0])
            assert pred in [0,1]
            preds.append(pred)

    elif eval_task == "reversed_video":
        batch_size = len(samples['text_input'])
        if eval_module == "backbone":
            compute_sim_func = task._compute_sim_backbone
        elif eval_module == "knowledge_patcher":
            if hasattr(model, "if_as_knowledge_fuser") and model.if_as_knowledge_fuser == True:
                compute_sim_func = task._compute_sim_main_fuser
            else:
                compute_sim_func = task._compute_sim_main
        elif eval_module == "knowledge_patcher_baseline":
            compute_sim_func = task._compute_sim_main_baseline
        preds = []
        targets = [0 for b in range(batch_size)]
        
        ### v1 ### 
        for b in range(batch_size):
            video_tensor = torch.stack((samples['video_input'][b],samples['video_input_reversed'][b]))
            text_cand = [samples["text_input"][b]]
            text_tensor = model.tokenize(text_cand).to(video_tensor.device)
            sims = compute_sim_func(model, text_tensor, video_tensor, v2t=False) # text -> vid (1, 2)
            assert sims.shape == (1,2)
            # handle the edge case where the model cannot distinguish reversed video by design, e.g., Image-based model without temporal embedding
            if abs(sims[0][0] - sims[0][1]) < 1e-5:
                pred = random.choice([0,1])
            else:
                pred = np.argmax(sims[0])
            assert pred in [0,1]
            preds.append(pred)

    elif eval_task == "video_text_matching":
        # return loss
        total_loss, losses = model(samples)
        return total_loss.item(), losses
    
    else:
        raise NotImplementedError("unknown task name")

    return sims, preds, targets

def inference_step_downstream_task_retrieval_v2t(task, model, text_feats, video_input, eval_task, eval_module):
    ### get video feature ###
    if eval_module == "backbone":
        video_feat = model.encode_video_backbone(video_input, return_all_feats=False) # (B, D)
        backbone_video_feat = video_feat
    elif eval_module == "knowledge_patcher_baseline":
        video_feat = model.encode_video(video_input, return_all_feats=False) # (B, D)
        backbone_video_feat = model.encode_video_backbone(video_input, return_all_feats=False)
    elif eval_module == "knowledge_patcher":
        backbone_video_feat, video_feat = model.encode_video(video_input) # (B, num_latents, D)
    else:
        raise NotImplementedError("unknown eval module type")
    video_feat = F.normalize(video_feat, dim=-1)
    backbone_video_feat = F.normalize(backbone_video_feat, dim=-1)
    
    ### get sim compute function ###
    if eval_module == "backbone":
        compute_sim_func = task._compute_sim_backbone
    elif eval_module == "knowledge_patcher":
        compute_sim_func = task._compute_sim_main_patcher_and_fuser
    elif eval_module == "knowledge_patcher_baseline":
        compute_sim_func = task._compute_sim_main_baseline

    sim_v2t = compute_sim_func(model, text_feats, video_feat, v2t=True)
    sim_v2t_backbone = compute_sim_func(model, text_feats, backbone_video_feat, v2t=True)
    print("sim_v2t.shape:", sim_v2t.shape) # v_len, t_len 
    print("sim_v2t_backbone.shape:", sim_v2t_backbone.shape) # v_len, t_len 
    return sim_v2t, sim_v2t_backbone

def inference_step_mcqa(task, model, samples, eval_task, eval_module):
    video_input = samples["video_input"] # B, num_frm, 3, 224, 224
    text_input = samples["text_input"] # list of list of string (B, 5)
    
    ## get video features
    if eval_module == "backbone":
        video_feat = model.encode_video_backbone(video_input, return_all_feats=False) # (B, D)
    elif eval_module == "knowledge_patcher_baseline":
        video_feat = model.encode_video(video_input, return_all_feats=False) # (B, D)
    elif eval_module == "knowledge_patcher":
        _, video_feat = model.encode_video(video_input) # (B, num_latents, D) or (B, D) if pooling or is fuser
    else:
        raise NotImplementedError("unknown eval module type")
    video_feat = F.normalize(video_feat, dim=-1)

    ## get text features
    # print("text_input:", text_input)
    text_tensor_batch = [model.tokenize(choices).to(model.device) for choices in text_input]
    text_feats_batch = []
    for text_tensor in text_tensor_batch:
        if eval_module == "backbone":
            raw_text_feat, perceiver_textual_embeddings = model.encode_text_backbone(text_tensor, return_all_feats=True)
        else:
            raw_text_feat, perceiver_textual_embeddings = model.encode_text(text_tensor)
        perceiver_textual_embeddings = F.normalize(perceiver_textual_embeddings, dim=-1)
        raw_text_feat = F.normalize(raw_text_feat, dim=-1)
        if hasattr(model, "text_perceiver") and model.text_perceiver is not None and eval_module != "backbone":
            text_feats_batch.append(perceiver_textual_embeddings) # sequence
        else:
            text_feats_batch.append(raw_text_feat) # single vector

    ## get similarity logits
    if eval_module == "backbone":
        compute_sim_func = task._compute_sim_backbone
    elif eval_module == "knowledge_patcher":
        compute_sim_func = task._compute_sim_main_patcher_and_fuser
    elif eval_module == "knowledge_patcher_baseline":
        compute_sim_func = task._compute_sim_main_baseline

    assert len(text_feats_batch) == video_feat.shape[0]
    
    preds_b = np.zeros(len(text_tensor_batch), dtype=int)
    targets_b = samples["answer"].detach().cpu().numpy()
    sim_v2t = []
    for i in range(len(text_tensor_batch)):
        v = video_feat[i].unsqueeze(0) # (1, D) or (1, num_latents, D)
        t = text_feats_batch[i] # (5, D) or (5, num_latents, D)
        sim_v2t_b = compute_sim_func(model, t, v, v2t=True)
        # print(sim_v2t_b)
        assert sim_v2t_b.shape == (1,5)
        pred = sim_v2t_b[0].argmax()
        preds_b[i] = pred
        sim_v2t.append(sim_v2t_b[0])

    return sim_v2t, preds_b, targets_b


def _reverse_normalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]):
    # reverse the normalization in vis_processor for visualization of the sampled frames
    reverse_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return reverse_normalize(tensor)

def save_video_frames(sample, output_dir, b=0):
    def get_concat_h(imgs):
        dst = Image.new('RGB', (sum([imgs[i].width for i in range(len(imgs))]), imgs[0].height))
        prev_img_end_idx = 0
        for i, img in enumerate(imgs):
            dst.paste(img, (prev_img_end_idx, 0))
            prev_img_end_idx += img.width
        return dst

    video_frames = sample['video_input'][b]            

    imgs = [transforms.ToPILImage()(_reverse_normalize(video_frames[i])) for i in range(len(video_frames))]
    get_concat_h(imgs).save(f"{output_dir}/{b}_original_frames.jpg")    
    get_concat_h(imgs[::-1]).save(f"{output_dir}/{b}_reversed_frames.jpg")

def ssv2_actionbench_inference(model, dataset, task, output_dir, num_query = 10, sample_ids=[]):
    print("dataset length:", len(dataset))
    if sample_ids == []:
        assert num_query > 0
        sample_ids = random.sample(range(len(dataset)), num_query)
    
    output_root = output_dir
    for sample_id in sample_ids:
        output_dir = os.path.join(output_root, str(sample_id))
        os.makedirs(output_dir, exist_ok=True)
        
        query_idx = None
        for i, ann in enumerate(dataset.annotation):
            if str(ann['clip_uid']) == str(sample_id):
                query_idx = i
                break
        assert query_idx is not None, "query_idx is None"
        
        input_sample = dataset[query_idx]

        # # overwrite the original text input:
        # print(input_sample)
        # input_sample['object_shuffled_text_input'] = 'Cellphone falling like a rock'

        input_sample = dataset.collater([input_sample]) # batch == 1

        # same sample video frames and annotation
        save_video_frames(input_sample, output_dir, b=0)

        # run inference
        eval_task = task.config.run_cfg.get("eval_task", "video_text_matching")
        print("eval_task:",eval_task)
        eval_module = task.config.run_cfg.get("eval_module", "backbone")
        print("eval_module:",eval_module)

        sims, preds, targets = inference_step_actionbench(task, model, input_sample, eval_task=eval_task, eval_module=eval_module)
        
        print("sims:", sims)
        print("preds:", preds)
        print("targets:", targets)

        # save annotation and scores
        sims = sims[0].tolist()
        if eval_task == "action_antonym":
            annotation_and_scores = {
                "text_input": input_sample['text_input'][0],
                "action_antonym_text_input": input_sample['action_antonym_text_input'][0],
                "eval_task":eval_task,
                "eval_module":eval_module,
                "scores":[
                    (input_sample['text_input'][0],sims[0]),
                    (input_sample['action_antonym_text_input'][0],sims[1]),
                ]
            }
        elif eval_task == "reversed_video":
            annotation_and_scores = {
                "text_input": input_sample['text_input'][0],
                "eval_task":eval_task,
                "eval_module":eval_module,
                "scores":[
                    ("original_video",sims[0]),
                    ("reversed_video",sims[1]),
                ]
            }
        elif eval_task == "object_shuffle":
            annotation_and_scores = {
                "text_input": input_sample['text_input'][0],
                "object_shuffled_text_input": input_sample['object_shuffled_text_input'][0],
                "eval_task":eval_task,
                "eval_module":eval_module,
                "scores":[
                    (input_sample['text_input'][0],sims[0]),
                    (input_sample['object_shuffled_text_input'][0],sims[1]),
                ]
            }

        print("annotation_and_scores:", annotation_and_scores)
        with open(f"{output_dir}/{eval_task}_result.json", 'w') as f:
            json.dump(annotation_and_scores, f, indent=4)

def downstream_task_v2t_retrieval_inference(model, dataset, task, output_dir, num_query = 10, sample_ids=[]):
    print("dataset length:", len(dataset))
    if sample_ids == []:
        assert num_query > 0
        sample_ids = random.sample(range(len(dataset)), num_query)
    
    model.eval()
    print("query sample_ids:", sample_ids)
    ### encode all text candidates ###
    print("computing text features...")
    backbone_text_feats = []
    text_feats = []
    
    texts = dataset.texts
    text_processor = dataset.text_processor
    text_bs = 16
    num_texts = len(texts)
    for i in range(0, num_texts, text_bs):
        text_cand = texts[i: min(num_texts, i+text_bs)]
        if text_processor is not None:
            text_cand = [text_processor(t) for t in text_cand]
        text_tensor_batch = model.tokenize(text_cand).to(model.device)
        # text_feats_batch = model.encode_text_backbone(text_tensor_batch)
        raw_text_feat, perceiver_textual_embeddings = model.encode_text(text_tensor_batch)
        if hasattr(model, "text_perceiver") and model.text_perceiver is not None and eval_module != "backbone":
            text_feats_batch = perceiver_textual_embeddings # sequence
        else:
            text_feats_batch = raw_text_feat # single vector
        # store model text feats
        text_feats_batch = F.normalize(text_feats_batch, dim=-1)
        text_feats.append(text_feats_batch)
        # store backbone text feats
        backbone_text_feat = F.normalize(raw_text_feat, dim=-1)
        backbone_text_feats.append(backbone_text_feat)
        
    text_feats = torch.cat(text_feats, dim=0) # (all_num_text, D) | (all_num_text, Q, D)
    # backbone_text_feats = torch.cat(backbone_text_feats, dim=0) # (all_num_text, D)
    print("text_feats.shape:", text_feats.shape)
    # print("backbone_text_feats.shape:", backbone_text_feats.shape)


    output_root = output_dir
    for sample_id in sample_ids:
        print("inferencing sample_id:", sample_id)
        query_idx = sample_id
        output_dir = os.path.join(output_root, str(query_idx))
        os.makedirs(output_dir, exist_ok=True)
        
        input_sample = dataset[query_idx]
        input_sample = dataset.collater([input_sample]) # batch == 1

        v2t_target = dataset.v2t_targets[query_idx]
        v2t_target_text = [dataset.texts[t] for t in v2t_target]
        
        assert input_sample['text_input'] == v2t_target_text

        # same sample video frames and annotation
        save_video_frames(input_sample, output_dir, b=0)

        # run inference
        eval_task = task.config.run_cfg.get("eval_task", "downstream_tasks_retrieval")
        print("eval_task:",eval_task)
        eval_module = task.config.run_cfg.get("eval_module", "backbone")
        print("eval_module:",eval_module)
        if_as_knowledge_fuser = task.config.model_cfg.get("if_as_knowledge_fuser", False)
        print("if_as_knowledge_fuser:",if_as_knowledge_fuser)

        ### compute v2t sim for the current video sample ###
        sim_v2t, sim_v2t_backbone = inference_step_downstream_task_retrieval_v2t(
            task, 
            model, 
            text_feats, 
            input_sample["video_input"], 
            eval_task=eval_task, 
            eval_module=eval_module
        )
        
        ### compute ranking ###
        score = sim_v2t[0]
        inds = np.argsort(score)[::-1]
        preds = []
        for i, ind in enumerate(inds):
            preds.append((float(score[ind]),texts[ind]))
            if ind in v2t_target:
                rank = i # the rank of the first item that is in gt targets

        # print("sim_v2t:", sim_v2t)
        # print("rank:", rank)
        # print("preds:", preds)
        # print("target:", v2t_target)
        # print("target_text:", v2t_target_text)

        # save annotation and scores
        annotation_and_scores = {
            "rank":rank,
            "target":v2t_target,
            "target_text":v2t_target_text,
            "preds":preds,
        }
        print("annotation_and_scores:", annotation_and_scores)
        if if_as_knowledge_fuser:
            eval_module = eval_module + "_knowledge_fuser"
        with open(f"{output_dir}/{eval_module}_result.json", 'w') as f:
            json.dump(annotation_and_scores, f, indent=4)

def downstream_task_next_qa(model, dataset, task, output_dir, num_query = 10, sample_ids=[]):
    print("dataset length:", len(dataset))
    if sample_ids == []:
        assert num_query > 0
        sample_ids = random.sample(range(len(dataset)), num_query)
    

    model.eval()
    output_root = output_dir
    for sample_id in sample_ids: 
        query_idx = sample_id
        output_dir = os.path.join(output_root, str(query_idx))
        os.makedirs(output_dir, exist_ok=True)
        
        input_sample = dataset[query_idx]
        input_sample = dataset.collater([input_sample]) # batch == 1

        # same sample video frames and annotation
        save_video_frames(input_sample, output_dir, b=0)

        # run inference
        eval_task = task.config.run_cfg.get("eval_task", "5way-multiple-choice-qa")
        print("eval_task:",eval_task)
        eval_module = task.config.run_cfg.get("eval_module", "backbone")
        print("eval_module:",eval_module)
        if_as_knowledge_fuser = task.config.model_cfg.get("if_as_knowledge_fuser", False)
        print("if_as_knowledge_fuser:",if_as_knowledge_fuser)

        
        # compute sim
        sim_v2t, preds, targets = inference_step_mcqa(task, model, input_sample, eval_task, eval_module)

        scores = sim_v2t[0].tolist()
        candidates = input_sample['text_input'][0]
        print(scores)
        print(candidates)
        assert len(scores) == len(candidates)
        cand_probs = [(s, c) for s, c in zip(scores, candidates)]
        cand_probs = sorted(cand_probs, key=lambda x: x[0], reverse=True)

        ### save annotation and scores ###
        annotation_and_scores = {
            "cand_probs":cand_probs,
            "target_text":candidates[int(targets[0])],
            # "pred":int(preds[0]),
            # "target":int(targets[0]),
        }
        print("annotation_and_scores:", annotation_and_scores)
        if if_as_knowledge_fuser:
            eval_module = eval_module + "_knowledge_fuser"
        with open(f"{output_dir}/{eval_module}_result.json", 'w') as f:
            json.dump(annotation_and_scores, f, indent=4)

def main(num_query = 10, sample_ids = []):
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()

    inference_type = args.inference_type
    print("inference_type:", inference_type)

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    # create task instance
    task = tasks.setup_task(cfg)

    # build dataset
    datasets = task.build_datasets(cfg)
    print("### successfully build datasets:", datasets)

    for dataset_name in datasets:
        for split, d in datasets[dataset_name].items():
            if split in ['val', 'test']:
                print(dataset_name, split, len(d))

    print("cfg.run_cfg.output_dir:", cfg.run_cfg.output_dir)
    
    # build model
    model = task.build_model(cfg)

    if inference_type == "actionbench":
        ssv2_actionbench_inference(
            model, 
            datasets['actionbench_ssv2_224x224_5fps']['val'], 
            task,
            output_dir=cfg.run_cfg.output_dir, 
            num_query=num_query,
            sample_ids=sample_ids
        )
    elif inference_type == "downstream_task_retrieval_v2t_temporal_ssv2":
        downstream_task_v2t_retrieval_inference(
            model, 
            datasets['downstream_tasks_temporal']['val'], 
            task,
            output_dir=cfg.run_cfg.output_dir, 
            num_query=num_query,
            sample_ids=sample_ids
        )
    elif inference_type in ["downstream_task_retrieval_v2t_ssv2_label", "downstream_task_retrieval_v2t_ssv2_template"]:
        downstream_task_v2t_retrieval_inference(
            model, 
            datasets['downstream_tasks_retrieval_ssv2_224x224_5fps']['val'], 
            task,
            output_dir=cfg.run_cfg.output_dir, 
            num_query=num_query,
            sample_ids=sample_ids
        )
    elif inference_type == "downstream_task_next_qa":
        downstream_task_next_qa(
            model, 
            datasets['downstream_tasks_qa_nextqa_224x224_5fps']['val'], 
            task, 
            output_dir=cfg.run_cfg.output_dir, 
            num_query=num_query,
            sample_ids=sample_ids
        )


if __name__ == "__main__":
    """usage:
        bash run_scripts/inference.sh 
    """
    random.seed(42) # Set the seed to 42
   
    ## set samples for inference
    num_query = 10
    sample_ids = [] # if sample_ids == [], randomly select num_query instances

    # # you can set custom sample ids by
    # num_query = 1
    # sample_ids = [204782]

    main(num_query, sample_ids)


