# %%
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import PIL
from glob import glob
import decord
from decord import VideoReader
import random
import logging

try:
    from lavis.datasets.datasets.base_dataset import BaseDataset
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.realpath(os.path.join(__file__, '../LAVIS')))
    from lavis.datasets.datasets.base_dataset import BaseDataset

from torch.utils.data.dataloader import default_collate
import pandas as pd
from collections import defaultdict
from typing import Literal, get_args, List

logging.getLogger().setLevel(logging.INFO)

decord.bridge.set_bridge('native')

### === AcDyBenchDataset === ###
class AcDyBenchDatasetBase(BaseDataset):
    def __init__(self,
        annotation,
        task = "video_text_matching",
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps=None,
        k=None,
        neg_sampling_same_clip=0
    ):
        """
            annotation: a list of annotation samples
        """
        self.task = task
        self.annotation = annotation
        print("annotation size:", len(self.annotation))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.frm_sampling_strategy = frm_sampling_strategy
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frm = num_frm
        self.fps = fps
        print("using fps:", self.fps)

        # if using negative sampling
        self.neg_sampling_same_clip = neg_sampling_same_clip
        if self.neg_sampling_same_clip > 0:
            print("Using negative sampling from same clip, number:", neg_sampling_same_clip)

    def _load_video_from_path_decord(self, video_path, fps, start_time=None, end_time = None):
        """
            returns frames as PIL images
        """
        frm_sampling_strategy = self.frm_sampling_strategy
        num_frm = self.num_frm
        task = self.task
        try:
            # load video
            if not self.frame_height or not self.frame_width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=self.frame_width, height=self.frame_height)

            # get start and end frame idx
            vlen = len(vr)
            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'
                start_idx = max(int(start_time * fps), 0)
                end_idx = min(int(end_time * fps), vlen-1)
                if start_idx == end_idx:
                    end_idx = min(start_idx + 1, vlen-1)
            else:
                start_idx, end_idx = 1, vlen-1

            # task

            if frm_sampling_strategy == 'uniform':
                frame_indices = np.linspace(
                    start_idx,
                    end_idx,
                    num = num_frm,
                    endpoint = True,
                    retstep = False,
                    dtype = int
                )
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices).asnumpy() # (num_frm, H, W, C)
            # raw_sample_frms = vr.get_batch(frame_indices) # tensor (num_frm, H, W, C)
            raw_sample_frms = [Image.fromarray(item, mode="RGB") for item in raw_sample_frms] # PIL Images

        except Exception as e:
            logging.info(e)
            return None
        # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # torch tensor
        # raw_sample_frms = transforms.ToPILImage()(raw_sample_frms).convert('RGB')
        return raw_sample_frms

    def collater(self, samples, additional_fields: List[str] = [], text_fields: List[str] = []):
        if self.vis_processor is None:
            return samples
        if self.neg_sampling_same_clip == 0:
            return default_collate(samples)
        else:
            # additional work for batching the positive and negative samples
            ret = {
                "video_input":[],
                "video_input_reversed":[],
                "video_input_before":[],
                "video_input_after":[],
                "text_input":[],
                "action_antonym_text_input":[],
                "text_input_before":[],
                "text_input_after":[],
                **{f : [] for f in additional_fields}
            }
            for sample in samples:
                pos_instance = sample['pos_instance']
                neg_instances = sample['neg_instances']
                for att in ret.keys():
                    if att in pos_instance:
                        ret[att] += [pos_instance[att]] + [neg[att] for neg in neg_instances]

            empty_field_keys = []
            for att in ret.keys():
                if att not in text_fields:
                    if ret[att] != []:
                        ret[att] = torch.stack(ret[att])
                if ret[att] == []:
                    empty_field_keys.append(att)
            for remove_key in empty_field_keys:
                del ret[remove_key]
            return ret

    def __getitem__(self, index):
        if self.neg_sampling_same_clip == 0:
            # if not using any negative sampling strategy
            return self._get_single_item(index)
        else:
            # additional work if using the negative sampling strategy from the same clip
            ann_instance = self.annotation[index]
            indices_from_same_clip = list(self.clipuid_2_idx[ann_instance['clip_uid']])
            indices_from_same_clip.remove(index)
            assert len(indices_from_same_clip) == len(self.clipuid_2_idx[ann_instance['clip_uid']]) - 1
            if len(indices_from_same_clip) == 0:
                # the clip has only this one instance, then randomly sample another instance as negative
                sampled_neg_indices = random.sample(list(range(len(self.annotation))), self.neg_sampling_same_clip)
                print("Warning: no other instances from the same clip can be sampled, randomly sample negative instances:", sampled_neg_indices)
            else:
                while len(indices_from_same_clip) < self.neg_sampling_same_clip:
                    indices_from_same_clip += indices_from_same_clip
                sampled_neg_indices = random.sample(indices_from_same_clip, self.neg_sampling_same_clip)

            pos_instance = self._get_single_item(index) # Dict
            neg_instances = [self._get_single_item(idx) for idx in sampled_neg_indices] # List[Dict]
            return {
                "pos_instance":pos_instance,
                "neg_instances":neg_instances
            }

class AcDyBenchDataset_Ego4D(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root,
        ann_path,
        state_change_filtering_json = None,
        state_change_filtering = False, # if true, use the filtered videos only during loading annotation
        task = "video_text_matching", # ["video_text_matching", "action_antonym", "object_shuffle", "reversed_video"]
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps=None,
        k=None,
        neg_sampling_same_clip=0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of EgoClip annotations
        params:
            vis_root (string): Root directory of video clips (e.g., datasets/Ego4D/video_clips/clips_downsampled_5fps_downsized_224x224)
            ann_path (string): path to the egoclip style ann jsonl, each line is one instance (e.g., AcDyBench/ego4d/egoclip_subset_action_antonyms_train_val_test_split/val.jsonl)
            state_change_filtering_json: path to the json file that contains indication of whether a video is state-change salient (e.g., AcDyBench/ego4d/egoclip_subset_action_antonyms_train_val_test_split/state_change_heavy_instance_filtering_val.json)
            task (string): what's the evaluation task: choose from ["video_text_matching", "action_antonym", "object_shuffle", "reversed_video"]
                - "video_text_matching": for general vl tasks; returns a video_input and the corresponding text
                - "action_antonym": returns a video_input and two text input: one original annotation text, one modified annotation text with all verbs repleced with their antonym
                - "object_shuffle": returns a video_input and two text input: one original annotation text, one modified annotation text with action's objects repleced with another random object
                - "reversed_video": returns two video input: an original video_input and a reversed video_input with all frames reversed; and a text input of the original text annotation
                Note: all video_input returned are a sequence of sampled frames with size (num_frm, C, H, W)
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        # Ego4d nouns and verbs taxonomy size: see AcDyBench/ego4d/*.csv for the index mapping
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary, not used for now, could be used for fine-grained negative sampling
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary, not used for now, could be used for fine-grained negative sampling

        self.vis_root = vis_root
        if isinstance(ann_path, list):
            ann_path = ann_path[0]

        self.clipuid_2_idx = defaultdict(list)

        if state_change_filtering_json is not None:
            self.state_change_filtering_dict = json.load(open(state_change_filtering_json))
        else:
            self.state_change_filtering_dict = None

        ann_full = []
        with open(ann_path) as f:
            for line in f:
                ann_instance = json.loads(line)
                clip_uid = ann_instance['clip_uid']
                clip_relative_start = ann_instance['clip_relative_start']
                clip_relative_end = ann_instance['clip_relative_end']
                # filter out missing video clips
                if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
                    # check if filter
                    if self.state_change_filtering_dict is not None and state_change_filtering:
                        key_ = clip_uid + "__" + str(clip_relative_start) + "__" + str(clip_relative_end)
                        if self.state_change_filtering_dict[key_]["if_keep"]:
                            self.clipuid_2_idx[clip_uid].append(len(ann_full))
                            ann_full.append(ann_instance)
                    else:
                        self.clipuid_2_idx[clip_uid].append(len(ann_full))
                        ann_full.append(ann_instance)
                    
                    if k is not None and len(ann_full) >= k:
                        break

        super().__init__(
            ann_full,
            task = task, # ["video_text_matching", "action_antonym", "object_shuffle", "reversed_video"]
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps=fps,
            k=k,
            neg_sampling_same_clip=neg_sampling_same_clip
        )

        print("annotation size:", len(self.annotation))

    def _get_noun_verb_vec(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        if 'tag_noun' in sample and 'tag_verb' in sample:
            noun_idx = eval(sample['tag_noun'])
            verb_idx = eval(sample['tag_verb'])
            for i in noun_idx:
                noun_vec[i] = 1
            for i in verb_idx:
                verb_vec[i] = 1
        else:
            noun_vec = None
            verb_vec = None
        return noun_vec, verb_vec # if output zero tensor, meaning not used for deciding loss mask

    def _get_single_item(self, index):
        """otuput format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann_instance = self.annotation[index]
        narration_text = ann_instance['clip_text']

        clip_uid = ann_instance['clip_uid']
        video_path = os.path.join(self.vis_root, f"{clip_uid}.mp4")
        start_time = ann_instance['clip_relative_start']
        end_time = ann_instance['clip_relative_end']
        if self.fps is None:
            fps = ann_instance['clip_fps']
        else:
            fps = self.fps # overiding the original fps in ann

        # get state_change_saliency_flag
        key_ = clip_uid + "__" + str(start_time) + "__" + str(end_time)
        if self.state_change_filtering_dict is not None:
            state_change_saliency_flag = self.state_change_filtering_dict[key_]["if_keep"]
        else:
            state_change_saliency_flag = True
        state_change_saliency_flag = torch.tensor(state_change_saliency_flag)    

        # get video frames
        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            fps,
            start_time=start_time,
            end_time=end_time
        )

        noun_vec, verb_vec = self._get_noun_verb_vec(ann_instance) # potentially useful for negative sampling (Egoclip annotation specific)

        if raw_sampled_frames is None:
            print(f"ERROR: cannot loading video clip: {ann_instance['clip_uid']}; try load another random instance...")
            return self.__getitem__(index+1)

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            narration_text = self.text_processor(narration_text)

        # task specific output
        if self.task == "action_antonym":
            narration_text_action_antonym = ann_instance['action_antonym_clip_text']
            if self.text_processor is not None:
                narration_text_action_antonym = self.text_processor(narration_text_action_antonym)
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "action_antonym_text_input": narration_text_action_antonym,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
                "state_change_saliency_flag": state_change_saliency_flag
            }
        elif self.task == "object_shuffle":
            narration_text_object_shuffled = ann_instance['object_shuffled_clip_text']
            if self.text_processor is not None:
                narration_text_object_shuffled = self.text_processor(narration_text_object_shuffled)
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "object_shuffled_text_input": narration_text_object_shuffled,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
                "state_change_saliency_flag": state_change_saliency_flag
            }
        elif self.task == "reversed_video":
            raw_sampled_frames_reverse = torch.flip(raw_sampled_frames, dims=[0])
            return {
                "video_input_reversed": raw_sampled_frames_reverse,
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
                "state_change_saliency_flag": state_change_saliency_flag
            }
        else: # "video_text_matching"
            # default output for general video-text matching tasks
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
                "state_change_saliency_flag": state_change_saliency_flag
            }

    ## overwrite collater here
    def collater(self, samples):
        return super().collater(
            samples,
            additional_fields=['noun_vec', 'verb_vec', "object_shuffled_text_input", "state_change_saliency_flag"],
            text_fields=["text_input","action_antonym_text_input","text_input_before","object_shuffled_text_input","text_input_after"]
        )

Split = Literal['train', 'val', 'test']
class AcDyBenchDataset_SSv2(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root: str,
        ann_root: str,
        split: Split,
        state_change_filtering = False, # if True, only keep the clips with state-change saliency
        use_templates_as_labels = False, # if True, use templates instead of the original annotations for text_input
        task = "video_text_matching", # ["video_text_matching", "action_antonym", "reversed_video"]
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm =  8,
        frame_height = None,
        frame_width = None,
        fps = None,
        k = None,
        neg_sampling_same_clip = 0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of SSv2
        params:
            vis_root (string): Root directory of video clips (e.g., datasets/SSv2/video_clips/clips_downsampled_5fps_downsized_224x224)
            ann_root (string): path to the annotations root directory (e.g., AcDyBench/ssv2/shuffled_object_and_action_antonyms)
            use_templates_as_labels (bool): Whether to output the raw label template instead of the in-filled template. The test set only has raw templates as annotations.
            task (string): what's the evaluation task: choose from ["video_text_matching", "action_antonym", "reversed_video"]
                - "video_text_matching": for general vl tasks; returns a video_input and the corresponding text
                - "action_antonym": returns a video_input and two text input: one original annotation text, one modified annotation text with all verbs repleced with their antonym
                - "reversed_video": returns two video input: an original video_input and a reversed video_input with all frames reversed; and a text input of the original text annotation
                Note: all video_input returned are a sequence of sampled frames with size (num_frm, C, H, W)
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        if split not in get_args(Split):
            raise ValueError(f'split must be one of {get_args(Split)}')

        if split == 'test' and not use_templates_as_labels:
            raise ValueError('Test set only has raw templates as annotations')

        self.split = split

        self.vis_root = vis_root
        self.ann_root = ann_root
        self.use_templates_as_labels = use_templates_as_labels

        self.templates_to_ints = self._load_templates_to_ints()

        self.task = task # _load_annotations needs this to be set
        self.state_change_filtering = state_change_filtering # _load_annotations needs this to be set
        annotations = self._load_annotations(split)

        self.clipuid_2_idx = {d['clip_uid'] : [i] for i, d in enumerate(annotations)} # Each clip has only one annotation

        self.neg_sampling_same_clip = 0 # set neg_sampling_same_clip to 0 for SSv2 since there is only one annotation per video

        super().__init__(
            annotations,
            task = task, # ["video_text_matching", "action_antonym", "reversed_video"]
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps = fps,
            k = k,
            neg_sampling_same_clip = 0
        )

    def _load_templates_to_ints(self):
        '''
        Loads mapping from templates to integers
        '''
        with open(os.path.join(self.ann_root, 'labels.json')) as f:
            j = json.load(f)

        return  {k : int(v) for k, v in j.items()}

    def _load_annotations(self, split: Split):
        '''
        Builds list of annotations each containing a video id, label, template, and placeholders
        from the annotation files.

        Args:
            split (Split): The stage to load the annotations for.

        Returns: A list of annotations with the following structure:
            [
                {
                    'clip_uid': int,
                    'label' : Optional[str],
                    'template' : str,
                    'placeholders' : Optional[List[str]]
                }
            ]
        '''
        # load train_antonym if exists; for look up action antonym for the original training set
        # for training instances that doesn't have an antonym, we will randomly pick another instance as action antonym text
        id_to_train_antonym_anns = {}
        train_antonym_path = os.path.join(self.ann_root, f'train_antonym.json')
        if os.path.exists(train_antonym_path):
            train_antonym_anns = json.load(open(train_antonym_path))
            for item in train_antonym_anns:
                id_to_train_antonym_anns[item['id']] = item

        if split in ['train', 'val']:
            # load state_change_filtering_json if any
            state_change_filtering_json_path = os.path.join(self.ann_root, f"state_change_heavy_instance_filtering_{split}.json")
            if os.path.exists(state_change_filtering_json_path):
                self.state_change_filtering_dict = json.load(open(state_change_filtering_json_path))
            else:
                self.state_change_filtering_dict = None

            # load main ann
            split = 'validation' if split == 'val' else 'train' # File is called "validation.json"

            with open(os.path.join(self.ann_root, f'{split}.json')) as f:
                j = json.load(f)

            annots = []
            for d in j:
                annot = {
                    'clip_uid': int(d['id']),
                    'label': d['label'].capitalize(),
                    'template': d['template'].replace('[something]', 'something'),
                    'placeholders': d['placeholders']
                }

                if self.state_change_filtering_dict is not None and self.state_change_filtering:
                    if not self.state_change_filtering_dict[str(d['id'])]['if_keep']:
                        continue

                if self.task == 'action_antonym':
                    if split == 'validation':
                        # assert that validation instances contains antonym items
                        annot.update({
                            'label_action_antonym_clip_text': d['label_action_antonym_clip_text'],
                            'template_action_antonym_clip_text': d['template_action_antonym_clip_text']
                        })
                    else: # split == 'train'
                        id_ = int(d['id'])
                        if id_ in id_to_train_antonym_anns:
                            # if current id has an action antonym
                            train_antonym_ann = id_to_train_antonym_anns[id_]
                            annot.update({
                                'label_action_antonym_clip_text': train_antonym_ann['label_action_antonym_clip_text'],
                                'template_action_antonym_clip_text': train_antonym_ann['template_action_antonym_clip_text']
                            })
                        else:
                            # if not, randomly pick an item in train_antonym, and using the original text as antonym
                            random_id = random.choice(list(id_to_train_antonym_anns.keys()))
                            train_antonym_ann = id_to_train_antonym_anns[random_id]
                            annot.update({
                                'label_action_antonym_clip_text': train_antonym_ann['label'],
                                'template_action_antonym_clip_text': train_antonym_ann['template']
                            })

                if self.task == 'object_shuffle':
                    annot.update({
                        'label_object_shuffled_clip_text': d['label_object_shuffled_clip_text']
                    })

                annots.append(annot)
        else:
            # columns = ['id', 'template'] if self.task not in ['action_antonym','object_shuffle','reversed_video'] else None # Infer from csv if action_antonym
            # df = pd.read_csv(os.path.join(self.ann_root, 'test-answers.csv'), names=columns, sep=';')
            
            df = pd.read_csv(os.path.join(self.ann_root, 'test-answers.csv'), header=0, sep=';')

            annots = []
            for r in df.itertuples():
                annot = {
                    'clip_uid': int(r.id),
                    'label': None,
                    'template': r.template,
                    'placeholders': None
                }

                if self.task == 'action_antonym':
                    annot.update({
                        'label_action_antonym_clip_text': None,
                        'template_action_antonym_clip_text': r.template_action_antonym_clip_text
                    })

                annots.append(annot)

        return annots

    def _get_single_item(self, index):
        """output format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann = self.annotation[index]
        narration_text = ann['template'] if self.use_templates_as_labels else ann['label']

        id = ann['clip_uid']

        # get state_change_saliency_flag
        if self.state_change_filtering_dict is not None:
            state_change_saliency_flag = self.state_change_filtering_dict[str(id)]['if_keep']
        else:
            state_change_saliency_flag = True
        state_change_saliency_flag = torch.tensor(state_change_saliency_flag)

        # get video frames
        video_path = os.path.join(self.vis_root, f"{id}.mp4")
        start_time = None # ann_instance['clip_relative_start']
        end_time = None # ann_instance['clip_relative_end']

        assert os.path.exists(video_path)
        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            self.fps,
            start_time=start_time,
            end_time=end_time
        )

        if raw_sampled_frames is None:
            print(f"ERROR: failed to load video clip with uid: {id}; trying to load another random instance...")
            return self.__getitem__(index+1)
        else:
            logging.debug(f'Successfully loaded clip with uid: {id}')

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            narration_text = self.text_processor(narration_text)

        # task specific output
        if self.task == "action_antonym":
            narration_text_action_antonym = ann[f'{"template" if self.use_templates_as_labels else "label"}_action_antonym_clip_text']
            if self.text_processor is not None:
                narration_text_action_antonym = self.text_processor(narration_text_action_antonym)
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "action_antonym_text_input": narration_text_action_antonym,
                "state_change_saliency_flag": state_change_saliency_flag
            }

        elif self.task == "object_shuffle":
            assert not self.use_templates_as_labels
            narration_text_object_shuffled = ann['label_object_shuffled_clip_text']
            if self.text_processor is not None:
                narration_text_object_shuffled = self.text_processor(narration_text_object_shuffled)
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "object_shuffled_text_input": narration_text_object_shuffled,
                "state_change_saliency_flag": state_change_saliency_flag
            }
        
        elif self.task == "reversed_video":
            raw_sampled_frames_reverse = torch.flip(raw_sampled_frames, dims=[0])
            return {
                "video_input_reversed": raw_sampled_frames_reverse,
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "state_change_saliency_flag": state_change_saliency_flag
            }
        else:
            # default output for general video-text matching tasks
            return {
                "video_input": raw_sampled_frames,
                "text_input": narration_text,
                "state_change_saliency_flag": state_change_saliency_flag
            }

    def collater(self, samples):
        return super().collater(
            samples,
            additional_fields=["object_shuffled_text_input", "state_change_saliency_flag"],
            text_fields=["text_input","action_antonym_text_input","text_input_before","object_shuffled_text_input","text_input_after"]
        )
        # return super().collater(
        #     samples,
        #     text_fields=["text_input","action_antonym_text_input","text_input_before","text_input_after"]
        # )


### === Downstream Datasets === ###
class DownstreamTask_Retrieval_SSv2(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root: str,
        ann_root: str,
        split: Literal['train', 'val'],
        task = Literal['ssv2_label','ssv2_template'],
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps = None,
        k = None,
        neg_sampling_same_clip = 0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of SSv2 for downstream retrieval tasks
        params:
            vis_root (string): Root directory of video clips (e.g., datasets/SSv2/video_clips/clips_downsampled_5fps_downsized_224x224)
            ann_root (string): path to the annotations root directory (e.g., datasets/SSv2/ssv2_label_ssv2_template)
            use_templates_as_labels (bool): Whether to output the raw label template instead of the in-filled template. The test set only has raw templates as annotations.
            task (string): ssv2_label or ssv2_template as described in https://arxiv.org/abs/2206.03428
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        self.split = split

        self.vis_root = vis_root
        self.ann_root = ann_root
        
        # self.templates_to_ints = self._load_templates_to_ints()

        self.task = task # _load_annotations needs this to be set
        
        (
            annotations, 
            self.videos, 
            self.texts, 
            self.v2t_targets,
            self.t2v_targets 
        ) = self._load_annotations(split)

        self.clipuid_2_idx = {d['clip_uid'] : [i] for i, d in enumerate(annotations)} # Each clip has only one annotation

        self.neg_sampling_same_clip = 0 # set neg_sampling_same_clip to 0 for SSv2 since there is only one annotation per video

        super().__init__(
            annotations,
            task = task,
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps = fps,
            k = k,
            neg_sampling_same_clip = 0
        )

    def _load_annotations(self, split):
        if split == "train" and self.task == "ssv2_label":
            ann_path = os.path.join(self.ann_root,"ssv2_ret_label_train.json")
        elif split == "val" and self.task == "ssv2_label":
            ann_path = os.path.join(self.ann_root,"ssv2_ret_label_val_small.json")
        elif split == "train" and self.task == "ssv2_template":
            ann_path = os.path.join(self.ann_root,"ssv2_ret_template_train.json")
        elif split == "val" and self.task == "ssv2_template":
            ann_path = os.path.join(self.ann_root,"ssv2_ret_template_val_small.json")
        
        j = json.load(open(ann_path))

        annots = []
        videos = [] 
        texts = []
        v2t_targets = []
        t2v_targets = []
        for item in j:
            video_names = item['video']
            if isinstance(video_names, str):
                video_names = [video_names]
            caption = item['caption'].replace('[something]', 'something')
            gt_video_idx_this_caption = []
            for video_name in video_names:
                video_id = video_name[:-5]
                if not self._check_if_video_exists(video_id):
                    logging.info("!!!! video {} does not exist, skip...".format(video_id))
                    continue
                # if video exists
                gt_video_idx_this_caption.append(len(videos)) # add current video idx
                videos.append(video_id)
                v2t_targets.append([len(texts)]) # one video only have one gt text
                annot = {
                    'clip_uid': int(video_id),
                    'caption': caption
                }
                annots.append(annot)
            texts.append(caption)
            t2v_targets.append(gt_video_idx_this_caption)
        
        assert len(videos) == len(v2t_targets)
        assert len(texts) == len(t2v_targets)
        return annots, videos, texts, v2t_targets, t2v_targets

    def _check_if_video_exists(self, video_id):
        return os.path.exists(os.path.join(self.vis_root, f"{video_id}.mp4"))

    def _get_single_item(self, index):
        """output format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann = self.annotation[index]
        narration_text = ann['caption']

        id = ann['clip_uid']
        video_path = os.path.join(self.vis_root, f"{id}.mp4")
        start_time = None # ann_instance['clip_relative_start']
        end_time = None # ann_instance['clip_relative_end']

        assert os.path.exists(video_path)
        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            self.fps,
            start_time=start_time,
            end_time=end_time
        )

        if raw_sampled_frames is None:
            print(f"ERROR: failed to load video clip with uid: {id}; trying to load another random instance...")
            return self.__getitem__(index+1)
        else:
            logging.debug(f'Successfully loaded clip with uid: {id}')

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            narration_text = self.text_processor(narration_text)

        # task specific output
        return {
            "video_input": raw_sampled_frames,
            "text_input": narration_text,
            "idx": int(index),
        }
    
    def collater(self, samples):
        return super().collater(
            samples,
            additional_fields=["idx"],
            text_fields=["text_input"]
        )

class DownstreamTask_MomentsInTime(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root: str,
        ann_root: str,
        split: Literal['train', 'val'],
        task = Literal['video_action_retrieval'],
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps = None,
        k = None,
        neg_sampling_same_clip = 0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of SSv2 for downstream retrieval tasks
        params:
            vis_root (string): Root directory of video clips (e.g., datasets/Moments_In_Time/videos)
            ann_root (string): path to the annotations root directory (e.g., datasets/Moments_In_Time/ann)
            task (string): video to action label
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        self.split = "training" if split == "train" else "validation"
        self.vis_root = os.path.join(vis_root, self.split)
        self.ann_root = ann_root
        
        self.task = task # only one task for now
        
        (
            annotations, 
            self.videos, 
            self.texts, 
            self.v2t_targets,
            self.t2v_targets 
        ) = self._load_annotations()

        self.clipuid_2_idx = {d['clip_uid'] : [i] for i, d in enumerate(annotations)} # Each clip has only one annotation

        self.neg_sampling_same_clip = 0 # set neg_sampling_same_clip to 0 for SSv2 since there is only one annotation per video

        super().__init__(
            annotations,
            task = task,
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps = fps,
            k = k,
            neg_sampling_same_clip = 0
        )

    def _load_annotations(self):
        if self.split == "training":
            ann_path = os.path.join(self.ann_root, "trainingSet.csv")
        else:
            ann_path = os.path.join(self.ann_root, "validationSet.csv")

        # get label texts & index mapping
        label_txt = os.path.join(self.ann_root, "moments_categories.txt")
        self.text_2_index = {}
        texts = []
        with open(label_txt, 'r') as f:
            for line in f:
                l, idx = line.split(',')
                l = l.strip().replace('+',' ')
                texts.append(l)
                self.text_2_index[l] = int(idx.strip())
        self.index_2_text = {value:key for key,value in self.text_2_index.items()}

        annotations = []
        videos = []
        v2t_targets = [] 
        t2v_targets = [[] for i in range(len(texts))]

        with open(ann_path, 'r') as f:
            for line in f:
                video_name, label, res_1, res_2 = line.split(',')
                vid_path = os.path.join(self.vis_root, video_name)
                if os.path.exists(vid_path):
                    text = label.strip().replace('+',' ')
                    annotations.append(
                        {
                            'clip_uid': video_name,
                            'caption': text
                        }
                    )
                    assert text in self.text_2_index
                    t_idx = self.text_2_index[text]
                    v2t_targets.append([t_idx])
                    t2v_targets[t_idx].append(len(videos))
                    videos.append(video_name)

        assert len(videos) == len(v2t_targets)
        assert len(texts) == len(t2v_targets)
        return annotations, videos, texts, v2t_targets, t2v_targets


    def _get_single_item(self, index):
        """output format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann = self.annotation[index]
        narration_text = ann['caption']
        label_index = self.text_2_index[narration_text]

        video_path = os.path.join(self.vis_root, ann['clip_uid'])
        start_time = None # ann_instance['clip_relative_start']
        end_time = None # ann_instance['clip_relative_end']

        assert os.path.exists(video_path)
        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            self.fps,
            start_time=start_time,
            end_time=end_time
        )

        if raw_sampled_frames is None:
            print(f"ERROR: failed to load video clip with uid: {id}; trying to load another random instance...")
            return self.__getitem__(index+1)
        else:
            logging.debug(f'Successfully loaded clip with uid: {id}')

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            narration_text = self.text_processor(narration_text)

        # task specific output
        return {
            "video_input": raw_sampled_frames,
            "text_input": narration_text,
            "label_index": label_index,
            "idx": int(index),
        }
    
    def collater(self, samples):
        return super().collater(
            samples,
            additional_fields=["idx","label_index"],
            text_fields=["text_input"]
        )

class DownstreamTask_Temporal(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root: dict,
        ann_root: str,
        split: Literal['val'],
        task = Literal['v1.0','v1.0_2.4k'],
        subset = Literal['kinetics','ssv2'],
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps = None,
        k = None,
        neg_sampling_same_clip = 0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of SSv2 for downstream retrieval tasks
        params:
            vis_root (dict): for example {
                'kinetics':'datasets/Temporal/video_clips/kinetics400/clips_downsampled_5fps_downsized_224x224', 
                'ssv2': 'datasets/SSv2/video_clips/clips_downsampled_5fps_downsized_224x224'
            }
            ann_root (string): path to the annotations root directory (e.g. datasets/Temporal/ann)
            task (string): provided two versions of val set https://arxiv.org/abs/2301.02074
            subset: a list specifing the sub-datasets to use ['kinetic', 'ssv2'] or ['kinetic'] or ['ssv2']
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        self.split = split
        self.vis_root = vis_root[subset]
        self.ann_root = ann_root

        self.subset = subset # _load_annotations needs this to be set
        self.task = task # _load_annotations needs this to be set
        
        (
            annotations, 
            self.videos, 
            self.texts, 
            self.v2t_targets,
            self.t2v_targets 
        ) = self._load_annotations(split)

        self.clipuid_2_idx = {d['clip_uid'] : [i] for i, d in enumerate(annotations)} # Each clip has only one annotation

        self.neg_sampling_same_clip = 0 # set neg_sampling_same_clip to 0 for SSv2 since there is only one annotation per video

        super().__init__(
            annotations,
            task = task,
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps = fps,
            k = k,
            neg_sampling_same_clip = 0
        )

    def _load_annotations(self, split):
        if split == "val":
            if self.task == "v1.0":
                ann_path = os.path.join(self.ann_root, "val-v1.0.csv")
            elif self.task == "v1.0_2.4k":
                ann_path = os.path.join(self.ann_root, "val-v1.0-2.4k.csv")
        
            annots = []
            videos = [] 
            texts = []

            df = pd.read_csv(ann_path, header=0)
            
            text_2_video_idx = defaultdict(list)
            for index, row in df.iterrows():
                dataset_name = row['dataset'].lower()
                if dataset_name != self.subset:
                    continue
                if self.subset == "ssv2":
                    video_id = row['video_id'].strip()
                    start_time = None
                    end_time = None
                else:
                    video_id = row['video_id'].split("/")[1].strip()[:-14]
                    time_string = row['video_id'].split("/")[1].strip()[-13:] 
                    start_time = float(time_string.split('_')[0])
                    end_time = float(time_string.split('_')[1])
                
                if not self._check_if_video_exists(video_id):
                    logging.info("!!!! video {} does not exist, skip...".format(video_id))
                    continue
                text = row['text'].replace('[something]', 'something')
                text_2_video_idx[text].append(len(videos))
                videos.append(video_id)
                annots.append(
                    {
                        'clip_uid': video_id,
                        'caption': text,
                        'start_time': start_time,
                        'end_time': end_time,
                    }
                )

            v2t_targets = [[] for i in range(len(videos))]
            t2v_targets = []
            for text,video_indices in text_2_video_idx.items():
                for v in video_indices:
                    v2t_targets[v] = [len(texts)]
                t2v_targets.append(video_indices)
                texts.append(text)
            
            assert len(videos) == len(v2t_targets)
            assert len(texts) == len(t2v_targets)
            return annots, videos, texts, v2t_targets, t2v_targets
        else:
            raise NotImplementedError("Temporal dataset for training not implemented")

    def _check_if_video_exists(self, video_id):
        return os.path.exists(os.path.join(self.vis_root, f"{video_id}.mp4"))

    def _get_single_item(self, index):
        """output format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann = self.annotation[index]
        narration_text = ann['caption']
        start_time = ann['start_time']
        end_time = ann['end_time']

        id = ann['clip_uid']
        video_path = os.path.join(self.vis_root, f"{id}.mp4")
        assert os.path.exists(video_path)

        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            self.fps,
            start_time=start_time,
            end_time=end_time
        )

        if raw_sampled_frames is None:
            print(f"ERROR: failed to load video clip with uid: {id}; trying to load another random instance...")
            return self.__getitem__(index+1)
        else:
            logging.debug(f'Successfully loaded clip with uid: {id}')

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            narration_text = self.text_processor(narration_text)

        # task specific output
        return {
            "video_input": raw_sampled_frames,
            "text_input": narration_text,
            "idx": int(index),
        }
    
    def collater(self, samples):
        return super().collater(
            samples,
            additional_fields=["idx"],
            text_fields=["text_input"]
        )

class DownstreamTask_QA_NextQA(AcDyBenchDatasetBase):
    def __init__(self,
        vis_root: dict,
        ann_root: str,
        split: Literal['train','val','test'],
        task = "mc", # multiple choice QA
        vis_processor = None,
        text_processor = None,
        frm_sampling_strategy = "uniform",
        num_frm = 8,
        frame_height = None,
        frame_width = None,
        fps = None,
        k = None,
        neg_sampling_same_clip = 0
    ):
        """
        dataset for loading video frames from video clips and preprocessed subset of SSv2 for downstream retrieval tasks
        params:
            vis_root (dict): path to videos (datasets/NextQA/video_clips/NExTVideo_downsampled_5fps_downsized_224x224)
            ann_root (string): path to the annotations root directory (datasets/NextQA/ann/nextqa)
            split: train, val, test
            vis_processor (Object class in processor.py): visual processor: add data augmentation and converting raw_sampled_frames (list of PIL images) returned from `self._load_video_from_path_decord` -> tensor
            text_processor (Object class in processor.py): textual processor: some basic processing of the input text (text string -> text string)
            frm_sampling_strategy (string): how do we sample frames from the clip: ['uniform']
            k (int): if not none, take a subset of k instances
        """
        self.split = split
        self.vis_root = vis_root
        self.ann_root = ann_root
        
        self.map_vid_vidorID = json.load(open(os.path.join(self.ann_root, "map_vid_vidorID.json")))
        annotations, clipuid_2_idx = self._load_annotations(split)

        self.clipuid_2_idx = clipuid_2_idx # each clip_uid can have multiple annotations

        super().__init__(
            annotations,
            task = task,
            vis_processor = vis_processor,
            text_processor = text_processor,
            frm_sampling_strategy = frm_sampling_strategy,
            num_frm = num_frm,
            frame_height = frame_height,
            frame_width = frame_width,
            fps = fps,
            k = k,
            neg_sampling_same_clip = neg_sampling_same_clip
        )

    def _load_annotations(self, split):
        csv = os.path.join(self.ann_root, f"{split}.csv")
        df = pd.read_csv(csv, header=0)
        
        annots = []
        clipuid_2_idx = defaultdict(list)
        for index, row in df.iterrows():
            vid = str(row['video'])
            if self._check_if_video_exists(vid):
                clipuid_2_idx[vid].append(len(annots))
                annots.append(
                    {
                        'clip_uid': vid,
                        'question': row['question'],
                        'answer': int(row['answer']),
                        'choices': [row[a] for a in ['a0','a1','a2','a3','a4']],
                        'action_antonym_choices': eval(row['action_antonym_choices']),
                        'qid': int(row['qid']),
                        'type': row['type'],
                        'start_time': None,
                        'end_time': None,
                    }
                )
        return annots, clipuid_2_idx 

    def _check_if_video_exists(self, video_id):
        return os.path.exists(os.path.join(self.vis_root, self.map_vid_vidorID[video_id] + ".mp4"))

    def _get_single_item(self, index):
        """output format:
            - video: tensor (num_frm, C, H, W)
            - text: string
        """
        ann = self.annotation[index]
        concat_qa_text = [ann['question']+"? "+ ans for ans in ann['choices']]
        action_antonym_concat_qa_text = [ann['question']+"? "+ ans for ans in ann['action_antonym_choices']]
        answer = ann['answer']
        start_time = ann['start_time']
        end_time = ann['end_time']

        id = ann['clip_uid']
        video_path = os.path.join(self.vis_root, self.map_vid_vidorID[id] + ".mp4")
        assert os.path.exists(video_path)

        raw_sampled_frames = self._load_video_from_path_decord(
            video_path,
            self.fps,
            start_time=start_time,
            end_time=end_time
        )

        if raw_sampled_frames is None:
            print(f"ERROR: failed to load video clip with uid: {id}; trying to load another random instance...")
            return self.__getitem__(index+1)
        else:
            logging.debug(f'Successfully loaded clip with uid: {id}')

        # preprocess frames and text
        if self.vis_processor is not None:
            try:
                raw_sampled_frames = self.vis_processor(raw_sampled_frames) # (num_frm, C, H, W)
            except:
                raw_sampled_frames = [self.vis_processor(f) for f in raw_sampled_frames] # (num_frm, C, H, W)
                raw_sampled_frames = torch.stack(raw_sampled_frames, dim=0)

        if self.text_processor is not None:
            concat_qa_text = [self.text_processor(t) for t in concat_qa_text]
        if self.text_processor is not None:
            action_antonym_concat_qa_text = [self.text_processor(t) for t in action_antonym_concat_qa_text]

        # task specific output
        return {
            "video_input": raw_sampled_frames,
            "text_input": concat_qa_text,
            "action_antonym_text_input": action_antonym_concat_qa_text,
            "answer": int(answer),
            "idx": int(index),
        }
    
    def collater(self, samples, text_fields = ['text_input', 'action_antonym_text_input']):
        if self.vis_processor is None:
            return samples
        ret = {
            "video_input":[],
            "text_input":[],
            "action_antonym_text_input":[],
            "answer": [],
            "idx": []
        }
        if self.neg_sampling_same_clip == 0:            
            for sample in samples:
                for att in ret.keys():
                    ret[att].append(sample[att])
        else:
            # additional work for batching the positive and negative samples
            for sample in samples:
                pos_instance = sample['pos_instance']
                neg_instances = sample['neg_instances']
                for att in ret.keys():
                    if att in pos_instance:
                        ret[att] += [pos_instance[att]] + [neg[att] for neg in neg_instances]
        empty_field_keys = []
        for att in ret.keys():
            if att not in text_fields:
                if ret[att] != []:
                    if not isinstance(ret[att][0], torch.Tensor):
                        ret[att] = [torch.tensor(item) for item in ret[att]]
                    ret[att] = torch.stack(ret[att])
            if ret[att] == []:
                empty_field_keys.append(att)
        for remove_key in empty_field_keys:
            del ret[remove_key]
        return ret