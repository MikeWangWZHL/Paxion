import json
import os
import torch.nn.functional as F

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample

import numpy as np
import random
import logging

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample

from einops import rearrange, repeat

from sklearn.metrics import precision_recall_fscore_support, f1_score

DEBUG = False

@registry.register_task("actionbench")
class PatchAndFuseBase(BaseTask):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def _compute_sim_backbone(self, model, text_tensor, video_tensor, v2t=True):
        text_features = model.encode_text_backbone(text_tensor) # (N, D)
        video_features = model.encode_video_backbone(video_tensor) # (M, D)
        video_features = torch.nn.functional.normalize(video_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        if hasattr(model.vl_backbone, "logit_scale"):
            t = model.vl_backbone.logit_scale.exp()
        elif hasattr(model.vl_backbone, "clipmodel"):
            t = model.vl_backbone.clipmodel.logit_scale.exp()
        elif hasattr(model.vl_backbone, "temp"):
            t = 1 / model.vl_backbone.temp
        else:
            t = 1.0

        if v2t:
            assert text_features.shape[0] > 1
            probs = (video_features @ text_features.T * t).softmax(dim=-1).detach().cpu().numpy() # (M, N)
        else:
            assert video_features.shape[0] > 1
            probs = (text_features @ video_features.T * t).softmax(dim=-1).detach().cpu().numpy() # (N, M)
        return probs

    def _compute_sim_main_baseline(self, model, text_tensor, video_tensor, v2t=True):
        text_features = model.encode_text(text_tensor, return_all_feats=False) # (N, D)
        video_features = model.encode_video(video_tensor, return_all_feats=False) # (M, D)
        video_features = torch.nn.functional.normalize(video_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        if hasattr(model, "temp"):
            t = model.temp
        else:
            t = 1.0
        if v2t:
            assert text_features.shape[0] > 1
            probs = (video_features @ text_features.T / t).softmax(dim=-1).detach().cpu().numpy() # (M, N)
        else:
            assert video_features.shape[0] > 1
            probs = (text_features @ video_features.T / t).softmax(dim=-1).detach().cpu().numpy() # (N, M)
        return probs

    def _compute_sim_main(self, model, text_tensor, video_tensor, v2t=True):
        text_features, percevier_textual_embeddings = model.encode_text(text_tensor) # (N, latent_dim)
        if text_features is None:
            text_features = percevier_textual_embeddings
        raw_visual_feature, perceiver_visual_embeddings = model.encode_video(
            video_tensor, 
            attn_guidance=None) # (M, latent_dim), (M, num_latents, latent_dim)
        perceiver_visual_features = torch.nn.functional.normalize(perceiver_visual_embeddings, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        if v2t:
            if text_features.dim() == 2:
                sim_q2t = torch.matmul(
                    perceiver_visual_features.unsqueeze(1), # (M, 1, num_query_tokens, latent_dim)  
                    text_features.unsqueeze(-1) # (N, latent_dim, 1)
                ).squeeze(-1)
                sim_v2t, _ = sim_q2t.max(-1) # value, indice
                sim_v2t = sim_v2t / model.temp # (M, N)
            else:
                sim_q2t = torch.matmul(
                    perceiver_visual_features.unsqueeze(1), # (M, 1, num_latents, latent_dim)  
                    rearrange(text_features, "n q d -> n d q") # (N, latent_dim, num_latents)
                ) # (M, N, num_latents, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_v2t, _ = sim_q2t.mean(dim=-1).max(dim=-1) # (batch_size, batch_size*num_gpu)
                sim_v2t = sim_v2t / model.temp # (M, N)
            assert text_features.shape[0] > 1
            probs = sim_v2t.softmax(dim=-1).detach().cpu().numpy() # (M, N)
        else:
            if text_features.dim() == 2:
                sim_t2q = torch.matmul(
                    text_features.unsqueeze(1).unsqueeze(1), # (N, 1, 1, latent_dim)
                    perceiver_visual_features.permute(0, 2, 1)  # (M, latent_dim, num_query_tokens)
                ).squeeze(-2)
                sim_t2v, _ = sim_t2q.max(-1)
                sim_t2v = sim_t2v / model.temp  # [batch_size, batch_size*num_gpu]
            else:
                sim_t2q = torch.matmul(
                    text_features.unsqueeze(1), # (N, 1, num_latents, latent_dim)  
                    rearrange(perceiver_visual_features, "m q d -> m d q") # (M, latent_dim, num_latents)
                ) # (N, M, num_latents, num_latents)
                # take mean across target tokens and then take max across all input tokens
                sim_t2v, _ = sim_t2q.mean(dim=-1).max(dim=-1) # (N, M)
                sim_t2v = sim_t2v / model.temp # (N, M)
            assert perceiver_visual_features.shape[0] > 1
            probs = sim_t2v.softmax(dim=-1).detach().cpu().numpy() # (N, M)
        return probs

    def _compute_sim_main_fuser(self, model, text_tensor, video_tensor, v2t=True):
        # assert fused_visual_features.dim() == 2
        text_features, percevier_textual_embeddings = model.encode_text(text_tensor) # (N, latent_dim)
        if text_features is None:
            text_features = percevier_textual_embeddings
        raw_visual_feature, fused_visual_features = model.encode_video(video_tensor) # (M, latent_dim), (M, num_latents, latent_dim) or (M, latent_dim)
        fused_visual_features = torch.nn.functional.normalize(fused_visual_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
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

    def train_step(self, model, samples):
        total_loss, losses = model(samples)
        return total_loss

    # NOTE: customized valid step
    def valid_step(self, model, samples, eval_task, eval_module):
        # NOTE: same as train step for pretraining
        if eval_task in ["action_antonym", "object_shuffle"]:
            batch_size = len(samples['text_input'])
            if eval_module == "backbone":
                compute_sim_func = self._compute_sim_backbone
            elif eval_module == "knowledge_patcher":                
                if hasattr(model, "if_as_knowledge_fuser") and model.if_as_knowledge_fuser == True:
                    compute_sim_func = self._compute_sim_main_fuser
                else:
                    compute_sim_func = self._compute_sim_main
            elif eval_module == "knowledge_patcher_baseline":
                compute_sim_func = self._compute_sim_main_baseline

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
                compute_sim_func = self._compute_sim_backbone
            elif eval_module == "knowledge_patcher":
                if hasattr(model, "if_as_knowledge_fuser") and model.if_as_knowledge_fuser == True:
                    compute_sim_func = self._compute_sim_main_fuser
                else:
                    compute_sim_func = self._compute_sim_main
            elif eval_module == "knowledge_patcher_baseline":
                compute_sim_func = self._compute_sim_main_baseline
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

        return preds, targets

    def evaluation(self, model, data_loader, cuda_enabled=True):
        
        eval_task = self.config.run_cfg.get("eval_task", "video_text_matching")
        print("eval_task:",eval_task)
        eval_module = self.config.run_cfg.get("eval_module", "backbone")

        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        if eval_task != "video_text_matching":
            results = {
                "preds":[],
                "targets":[]
            }
        else:
            results = {"val_loss":[]}

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            if eval_task != "video_text_matching":
                preds, targets = self.valid_step(
                    model=model, 
                    samples=samples, 
                    eval_task=eval_task,
                    eval_module=eval_module
                )
                results["preds"] += preds
                results["targets"] += targets
            else:
                val_loss, losses = self.valid_step(
                    model=model, 
                    samples=samples, 
                    eval_task=eval_task,
                    eval_module=eval_module
                )
                results["val_loss"].append(val_loss)

            if DEBUG:
                logging.info("!!!DEBUGGING: early break after 1 val iters.")
                break

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def after_evaluation(self, **kwargs):
        val_results = kwargs['val_result']
        split_name = kwargs['split_name']
        epoch = kwargs['epoch']

        eval_task = self.config.run_cfg.get("eval_task", "video_text_matching")
        
        if "preds" in val_results:
            preds = val_results['preds']
            targets = val_results['targets']

            assert len(preds) == len(targets)
            macro_f1 = f1_score(targets, preds, average='macro')
            micro_f1 = f1_score(targets, preds, average='micro')


            logging.info(
                "Epoch: {} | Split: {} | Task: {} | macro/micro: {}/{}".format(
                    epoch, split_name, eval_task, macro_f1, micro_f1
                )
            )
            # if having results from some metrics, return in the field: "agg_metrics"
            # if "agg_metrics" is returned, it will be used for deciding which checkpoint is better
            # otherwise, checkpoint with lower val_loss will be selected
            return {
                "task":eval_task,
                "agg_metrics":macro_f1 + micro_f1,
                "f1_scores":{
                    "macro_f1":macro_f1,
                    "micro_f1":micro_f1
                },
                "epoch":epoch,
                "split":split_name,
                "preds":[int(p) for p in preds],
                "targets":[int(t) for t in targets]
            }
        else:
            val_losses = val_results['val_loss']
            avg_val_loss = sum(val_losses) / len(val_losses)
            logging.info(
                "Epoch: {} | Split: {} | Task: {} | avg_val_loss: {}".format(
                    epoch, split_name, eval_task, avg_val_loss
                )
            )
            return {
                "task":eval_task,
                "val_loss": avg_val_loss,
                "epoch":epoch,
                "split":split_name            
            }

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        
        print('### dataset_config:', datasets_config)

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets
    
    def build_model(self, cfg):
        model_config = cfg.model_cfg
        print('### model_config:', model_config)
        model_cls = registry.get_model_class(model_config.arch)
        print('### model class:', model_cls)
        return model_cls.from_config(model_config)

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            if i == 0:
                print("sample input exmaple:", samples)
            
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if DEBUG and i >= 1:
                logging.info(f"DEBUGGING: early break after {i} training iters.")
                break

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

@registry.register_task("downstream_tasks_retrieval")
class DownstreamTaskRetrieval(PatchAndFuseBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

    def _compute_sim_backbone(self, model, text_features, video_features, v2t=True):
        
        if hasattr(model.vl_backbone, "logit_scale"):
            t = model.vl_backbone.logit_scale.exp()
        elif hasattr(model.vl_backbone, "clipmodel"):
            t = model.vl_backbone.clipmodel.logit_scale.exp()
        elif hasattr(model.vl_backbone, "temp"):
            t = 1 / model.vl_backbone.temp
        else:
            t = 1.0
        # print("temporature scaling:", t)

        if v2t:
            assert text_features.shape[0] > 1
            probs = (video_features @ text_features.T * t).softmax(dim=-1).detach().cpu().numpy() # (M, N)
        else:
            assert video_features.shape[0] > 1
            probs = (text_features @ video_features.T * t).softmax(dim=-1).detach().cpu().numpy() # (N, M)
        return probs

    def _compute_sim_main_baseline(self, model, text_features, video_features, v2t=True):
        t = model.temp
        if v2t:
            assert text_features.shape[0] > 1
            probs = (video_features @ text_features.T / t).softmax(dim=-1).detach().cpu().numpy() # (M, N)
        else:
            assert video_features.shape[0] > 1
            probs = (text_features @ video_features.T / t).softmax(dim=-1).detach().cpu().numpy() # (N, M)
        return probs

    def _compute_sim_main_patcher_and_fuser(self, model, text_features, fused_visual_features, v2t=True):
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

    @torch.no_grad()
    def evaluation(self, model, data_loader, cuda_enabled=True):
        ## using two step inference
        model.eval()

        eval_task = self.config.run_cfg.get("eval_task", "training")
        eval_module = self.config.run_cfg.get("eval_module", "backbone")
        eval_method = self.config.run_cfg.get("eval_method", "default")
        print("eval_task:",eval_task)
        print("eval_module:",eval_module)
        print("eval_method:",eval_method)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        ### encode all text candidates ###
        print("computing text features...")
        backbone_text_feats = []
        text_feats = []
        
        texts = data_loader.dataset.texts
        text_processor = data_loader.dataset.text_processor
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
        backbone_text_feats = torch.cat(backbone_text_feats, dim=0) # (all_num_text, D)
        print("text_feats.shape:", text_feats.shape)
        print("backbone_text_feats.shape:", backbone_text_feats.shape)

        ### encode video feats ###
        backbone_video_feats = []
        video_feats = []
        video_indices = []
        print("computing video features...")
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            video_input = samples["video_input"]
            video_idx = samples["idx"]
            video_indices.append(video_idx)
            # get video features
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
            # store model video feat
            video_feat = F.normalize(video_feat, dim=-1)
            video_feats.append(video_feat)

            # store backbone video feat
            backbone_video_feat = F.normalize(backbone_video_feat, dim=-1)
            backbone_video_feats.append(backbone_video_feat)
        
        video_feats = torch.cat(video_feats, dim=0)
        backbone_video_feats = torch.cat(backbone_video_feats, dim=0)
        video_indices = torch.cat(video_indices, dim=0)
        print("video_indices.shape", video_indices.shape)
        print("video_feats.shape", video_feats.shape)
        print("backbone_video_feats.shape", backbone_video_feats.shape)
        
        ### compute sim matrix ### 
        if eval_module == "backbone":
            compute_sim_func = self._compute_sim_backbone
        elif eval_module == "knowledge_patcher":
            # if hasattr(model, "if_as_knowledge_fuser") and model.if_as_knowledge_fuser == True:
            #     compute_sim_func = self._compute_sim_main_fuser
            # else:
            #     compute_sim_func = self._compute_sim_main
            compute_sim_func = self._compute_sim_main_patcher_and_fuser
        elif eval_module == "knowledge_patcher_baseline":
            compute_sim_func = self._compute_sim_main_baseline
        
        sim_v2t = []
        sim_v2t_backbone = []
        video_bs = 4
        num_videos = video_feats.shape[0]
        for i in range(0, num_videos, video_bs):
            # model feat sim
            video_feats_b = video_feats[i: min(num_videos, i+video_bs)]
            sim_v2t_b = compute_sim_func(model, text_feats, video_feats_b, v2t=True) # numpy array (v_len_b, full_t_len)
            sim_v2t.append(sim_v2t_b)
            # backbone feat sim
            backbone_video_feats_b = backbone_video_feats[i: min(num_videos, i+video_bs)]
            sim_v2t_backbone_b = self._compute_sim_backbone(model, backbone_text_feats, backbone_video_feats_b, v2t=True) # numpy array (v_len_b, full_t_len)
            sim_v2t_backbone.append(sim_v2t_backbone_b)
        sim_v2t = np.concatenate(sim_v2t, axis=0)
        sim_v2t_backbone = np.concatenate(sim_v2t_backbone, axis=0)
        assert sim_v2t.shape == sim_v2t_backbone.shape
        
        sim_t2v = []
        sim_t2v_backbone = []
        text_bs = 2
        for i in range(0, num_texts, text_bs):
            # model feat sim
            text_feats_b = text_feats[i: min(num_texts, i+text_bs)]
            sim_t2v_b = compute_sim_func(model, text_feats_b, video_feats, v2t=False) # numpy array (t_len_b, full_v_len)
            sim_t2v.append(sim_t2v_b)        
            # backbone feat sim
            backbone_text_feats_b = backbone_text_feats[i: min(num_texts, i+text_bs)]
            sim_t2v_backbone_b = self._compute_sim_backbone(model, backbone_text_feats_b, backbone_video_feats, v2t=False) # numpy array (t_len_b, full_v_len)
            sim_t2v_backbone.append(sim_t2v_backbone_b)        
        sim_t2v = np.concatenate(sim_t2v, axis=0)
        sim_t2v_backbone = np.concatenate(sim_t2v_backbone, axis=0)
        assert sim_t2v.shape == sim_t2v_backbone.shape
        print("sim_v2t.shape:", sim_v2t.shape) # v_len, t_len 
        print("sim_t2v.shape:", sim_t2v.shape) # t_len, v_len
        print("sim_v2t_backbone.shape:", sim_v2t_backbone.shape) # v_len, t_len 
        print("sim_t2v_backbone.shape:", sim_t2v_backbone.shape) # t_len, v_len

        if eval_method == "default":
            v2t_scores = sim_v2t
            t2v_scores = sim_t2v
        elif eval_method == "two_step_independent":
            top_k = 50
            print(f"using two_step_independent with top_k {top_k}")

            v2t_step1_topk_indices = np.argsort(-sim_v2t_backbone, axis=1)[:,:top_k]
            v2t_scores = np.zeros_like(sim_v2t_backbone)
            v2t_scores[np.arange(len(sim_v2t_backbone))[:, None], v2t_step1_topk_indices] \
                = sim_v2t[np.arange(len(sim_v2t_backbone))[:, None], v2t_step1_topk_indices]
            # v2t_scores[np.arange(len(sim_v2t_backbone))[:, None], v2t_step1_topk_indices] \
            #     = sim_v2t[np.arange(len(sim_v2t_backbone))[:, None], v2t_step1_topk_indices] + sim_v2t_backbone[np.arange(len(sim_v2t_backbone))[:, None], v2t_step1_topk_indices]

            t2v_step1_topk_indices = np.argsort(-sim_t2v_backbone, axis=1)[:,:top_k]
            t2v_scores = np.zeros_like(sim_t2v_backbone)
            t2v_scores[np.arange(len(sim_t2v_backbone))[:, None], t2v_step1_topk_indices] \
                = sim_t2v[np.arange(len(sim_t2v_backbone))[:, None], t2v_step1_topk_indices]
        elif eval_method == "ensemble_with_backbone":
            v2t_scores = sim_v2t + sim_v2t_backbone
            t2v_scores = sim_t2v + sim_t2v_backbone
        else:
            raise NotImplementedError


        v2t_targets = data_loader.dataset.v2t_targets # dict
        t2v_targets = data_loader.dataset.t2v_targets # dict
        print("v2t_targets:", v2t_targets)
        print("t2v_targets", t2v_targets)

        if eval_task != "training":
            results = {
                "sim_v2t":v2t_scores,
                "sim_t2v":t2v_scores,
                "v2t_targets":v2t_targets, # list of lists (gt ids)
                "t2v_targets":t2v_targets, # list of lists (gt ids)
                "video_indices":video_indices,
            }
        else:
            results = {"val_loss":[]}

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results
    
    def after_evaluation(self, **kwargs):
        val_results = kwargs['val_result']
        split_name = kwargs['split_name']
        epoch = kwargs['epoch']

        eval_task = self.config.run_cfg.get("eval_task", "training")
        eval_method = self.config.run_cfg.get("eval_method", "default")
        
        if "sim_v2t" in val_results:
            sim_v2t = val_results['sim_v2t']
            sim_t2v = val_results['sim_t2v']
            v2t_targets = val_results['v2t_targets']
            t2v_targets = val_results['t2v_targets']

            # v2t
            ranks = np.zeros(sim_v2t.shape[0])
            for index, score in enumerate(sim_v2t):
                inds = np.argsort(score)[::-1]
                targets = v2t_targets[index] # a list, can have multiple items
                for i, ind in enumerate(inds):
                    if ind in targets:
                        rank = i # the rank of the first item that is in gt targets
                        break
                ranks[index] = rank

            v2t_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            v2t_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            v2t_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

            # t2v
            ranks = np.zeros(sim_t2v.shape[0])
            for index, score in enumerate(sim_t2v):
                inds = np.argsort(score)[::-1]
                targets = t2v_targets[index] # a list, can have multiple items
                for i, ind in enumerate(inds):
                    if ind in targets:
                        rank = i # the rank of the first item that is in gt targets
                        break                
                ranks[index] = rank

            t2v_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            t2v_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            t2v_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

            logging.info(
                "Epoch: {} | Split: {} | Task: {} | v2t_r1: {}; v2t_r5: {}; t2v_r1: {}; t2v_r5: {}".format(
                    epoch, split_name, eval_task, v2t_r1, v2t_r5, t2v_r1, t2v_r5
                )
            )
            # if having results from some metrics, return in the field: "agg_metrics"
            # if "agg_metrics" is returned, it will be used for deciding which checkpoint is better
            # otherwise, checkpoint with lower val_loss will be selected
            
            v2t_r_mean = (v2t_r1 + v2t_r5 + v2t_r10) / 3
            t2v_r_mean = (t2v_r1 + t2v_r5 + t2v_r10) / 3
            r_mean = (v2t_r_mean + t2v_r_mean) / 2
            
            return {
                "task":eval_task,
                "method":eval_method,
                "v2t_r1":v2t_r1,
                "v2t_r5":v2t_r5,
                "v2t_r10":v2t_r10,
                "t2v_r1":t2v_r1,
                "t2v_r5":t2v_r5,
                "t2v_r10":t2v_r10,
                "agg_metrics":r_mean,
                "epoch":epoch,
                "split":split_name
            }
        else:
            val_losses = val_results['val_loss']
            avg_val_loss = sum(val_losses) / len(val_losses)
            logging.info(
                "Epoch: {} | Split: {} | Task: {} | avg_val_loss: {}".format(
                    epoch, split_name, eval_task, avg_val_loss
                )
            )
            return {
                "task":eval_task,
                "val_loss": avg_val_loss,
                "epoch":epoch,
                "split":split_name            
            }

@registry.register_task("downstream_tasks_multi_choice_qa")
class DownstreamTaskMCQA(DownstreamTaskRetrieval):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg
    
    @torch.no_grad()
    def evaluation(self, model, data_loader, cuda_enabled=True):
        model.eval()
        eval_module = self.config.run_cfg.get("eval_module", "backbone")
        print("eval_module:", eval_module)

        preds_all = []
        targets_all = []

        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
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
                compute_sim_func = self._compute_sim_backbone
            elif eval_module == "knowledge_patcher":
                compute_sim_func = self._compute_sim_main_patcher_and_fuser
            elif eval_module == "knowledge_patcher_baseline":
                compute_sim_func = self._compute_sim_main_baseline

            assert len(text_feats_batch) == video_feat.shape[0]
            
            preds_b = np.zeros(len(text_tensor_batch))
            targets_b = samples["answer"].detach().cpu().numpy()
            
            for i in range(len(text_tensor_batch)):
                v = video_feat[i].unsqueeze(0) # (1, D) or (1, num_latents, D)
                t = text_feats_batch[i] # (5, D) or (5, num_latents, D)
                sim_v2t_b = compute_sim_func(model, t, v, v2t=True)
                # print(sim_v2t_b)
                assert sim_v2t_b.shape == (1,5)
                pred = sim_v2t_b[0].argmax()
                preds_b[i] = pred
            
            preds_all.append(preds_b)
            targets_all.append(targets_b)
                
        preds_all = np.concatenate(preds_all, axis=0)
        targets_all = np.concatenate(targets_all, axis=0)
        assert preds_all.shape == targets_all.shape

        return {
            "preds": preds_all,
            "targets": targets_all
        }
    
    def after_evaluation(self, **kwargs):
        val_results = kwargs['val_result']
        split_name = kwargs['split_name']
        epoch = kwargs['epoch']

        eval_task = self.config.run_cfg.get("eval_task", "5way-multiple-choice-qa")
        
        assert "preds" in val_results
        assert "targets" in val_results

        preds = val_results['preds']
        targets = val_results['targets']

        assert len(preds) == len(targets)
        macro_f1 = f1_score(targets, preds, average='macro')
        micro_f1 = f1_score(targets, preds, average='micro')

        logging.info(
            "Epoch: {} | Split: {} | Task: {} | macro/micro: {}/{}".format(
                epoch, split_name, eval_task, macro_f1, micro_f1
            )
        )
        # if having results from some metrics, return in the field: "agg_metrics"
        # if "agg_metrics" is returned, it will be used for deciding which checkpoint is better
        # otherwise, checkpoint with lower val_loss will be selected
        return {
            "task":eval_task,
            "agg_metrics":macro_f1 + micro_f1,
            "f1_scores":{
                "macro_f1":macro_f1,
                "micro_f1":micro_f1
            },
            "epoch":epoch,
            "split":split_name,
            "preds":[int(p) for p in preds],
            "targets":[int(t) for t in targets]
        }