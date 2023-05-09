from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from data import (
    AcDyBenchDataset_Ego4D,
    AcDyBenchDataset_SSv2,
    DownstreamTask_Retrieval_SSv2,
    DownstreamTask_MomentsInTime,
    DownstreamTask_Temporal,
    DownstreamTask_QA_NextQA
)

import logging
import os
import shutil
import warnings

import lavis.common.utils as utils
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common.registry import registry
from lavis.datasets.data_utils import extract_archive
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url


@registry.register_builder("acdybench_ego4d_224x224_5fps")
class AcDyBenchDataset_Ego4D_Builder(BaseDatasetBuilder):
    train_dataset_cls = AcDyBenchDataset_Ego4D
    eval_dataset_cls = AcDyBenchDataset_Ego4D

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/acdybench/acdybench_ego4d_224x224_5fps.yaml",
        "object_shuffled": "configs/datasets/acdybench/acdybench_ego4d_object_shuffled_224x224_5fps.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()
        print(self.vis_processors)
        print(self.text_processors)

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        # task: ["video_text_matching", "action_antonym", "consequence_precondition", "reversed_video"]
        task = self.config.get('task', "action_antonym")
        fps = self.config.get('fps', None)

        # additional arguments

        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 4)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)

        state_change_filtering = self.config.get('state_change_filtering', False)
        print("## if use state_change_filtering:", state_change_filtering)

        datasets = dict()
        for split in ann_info.keys():

            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # state_change_filtering_json
            state_change_filtering_json = ann_info.get(split).get("state_change_filtering_json", None)
            print("use state_change_filtering_json:", state_change_filtering_json)
            
            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls

            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
                neg_sampling_same_clip = self.config.get('neg_sampling_same_clip', 0)
                print(f"## build training dataset with {neg_sampling_same_clip} neg samples per instance")
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                neg_sampling_same_clip = 0
                k = eval_k

            datasets[split] = dataset_cls(
                ann_path=ann_paths,
                vis_root=vis_path,
                state_change_filtering_json=state_change_filtering_json,
                state_change_filtering=state_change_filtering,
                task=task,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy = frm_sampling_strategy,
                num_frm = num_frm,
                frame_height = frame_height,
                frame_width = frame_width,
                fps=fps,
                k=k,
                neg_sampling_same_clip=neg_sampling_same_clip
            )

        return datasets

@registry.register_builder("acdybench_ssv2_224x224_5fps")
class AcDyBenchDataset_SSv2_Builder(BaseDatasetBuilder):

    train_dataset_cls = AcDyBenchDataset_SSv2
    eval_dataset_cls = AcDyBenchDataset_SSv2

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/acdybench/acdybench_ssv2_224x224_5fps.yaml",
        "action_antonyms_and_object_shuffled": "configs/datasets/acdybench/acdybench_ssv2_antonyms_224x224_5fps.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # TODO: add downloading function
        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        fps = self.config.get('fps', 5)

        
        # additional arguments
        state_change_filtering = self.config.get('state_change_filtering', False)
        print(f"## if use state_change_filtering:", state_change_filtering)
        
        use_templates_as_labels = self.config.get('use_templates_as_labels', False)

        neg_sampling_same_clip = 0
        print(f"## set neg_sampling_same_clip to 0 for SSv2 dataset")
        
        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 4)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)
        
        task = self.config.get('task', "action_antonym")
        print("## task:", task)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_root = ann_info.get(split).get('path')
            vis_root = build_info.get('videos').get('path')

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            

            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                k = eval_k
            
            if split == "test":
                use_templates_as_labels = True
            
            print(f"## if use template for split {split}: {use_templates_as_labels}")
            
            datasets[split] = dataset_cls(
                ann_root=ann_root,
                vis_root=vis_root,
                split=split,
                state_change_filtering=state_change_filtering,
                use_templates_as_labels=use_templates_as_labels,
                task=task,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy=frm_sampling_strategy,
                num_frm=num_frm,
                frame_height=frame_height,
                frame_width=frame_width,
                fps=fps,
                k=k,
                neg_sampling_same_clip=neg_sampling_same_clip
                # use_templates_as_labels=ann_info.get(split).get('use_templates_as_labels'),
            )

        return datasets
    

@registry.register_builder("downstream_tasks_retrieval_ssv2_224x224_5fps")
class DownstreamTasksRetrieval_SSv2_Builder(BaseDatasetBuilder):

    train_dataset_cls = DownstreamTask_Retrieval_SSv2
    eval_dataset_cls = DownstreamTask_Retrieval_SSv2

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/downstream_tasks/downstream_tasks_retrieval_ssv2_224x224_5fps.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # TODO: add downloading function
        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        fps = self.config.get('fps', 5)

        # additional arguments        
        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 8)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)
        
        task = self.config.get('task', 'ssv2_label')

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_root = ann_info.get(split).get('path')
            vis_root = build_info.get('videos').get('path')

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                k = eval_k

            datasets[split] = dataset_cls(
                vis_root=vis_root,
                ann_root=ann_root,
                split=split,
                task=task,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy=frm_sampling_strategy,
                num_frm=num_frm,
                frame_height=frame_height,
                frame_width=frame_width,
                fps=fps,
                k=k
            )

        return datasets

@registry.register_builder("downstream_tasks_moment_in_time")
class DownstreamTasks_MomentInTime_Builder(BaseDatasetBuilder):

    train_dataset_cls = DownstreamTask_MomentsInTime
    eval_dataset_cls = DownstreamTask_MomentsInTime

    DATASET_CONFIG_DICT = {
        "default": "datasets/downstream_tasks/downstream_tasks_moments_in_time.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # TODO: add downloading function
        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        fps = self.config.get('fps', None) # None for not using start and end time

        # additional arguments        
        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 8)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)
        
        task = self.config.get('task', 'video_action_retrieval')

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_root = ann_info.get(split).get('path')
            vis_root = build_info.get('videos').get('path')

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                k = eval_k

            datasets[split] = dataset_cls(
                vis_root=vis_root,
                ann_root=ann_root,
                split=split,
                task=task,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy=frm_sampling_strategy,
                num_frm=num_frm,
                frame_height=frame_height,
                frame_width=frame_width,
                fps=fps,
                k=k
            )

        return datasets

@registry.register_builder("downstream_tasks_temporal")
class DownstreamTasks_Temporal_Builder(BaseDatasetBuilder):

    train_dataset_cls = DownstreamTask_Temporal
    eval_dataset_cls = DownstreamTask_Temporal

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/downstream_tasks/downstream_tasks_temporal_224x224_5fps.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # TODO: add downloading function
        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        fps = self.config.get('fps', None) # None for not using start and end time

        # additional arguments        
        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 8)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)
        
        task = self.config.get('task', 'v1.0_2.4k')
        subset = self.config.get('subset', 'kinetics')
        print("## using subset:", subset)
        
        if subset == 'kinetics':
            assert fps is not None, "fps has to be specified for kinetics"

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_root = ann_info.get(split).get('path')
            vis_root = build_info.get('videos').get('path')
            print(vis_root)

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                k = eval_k

            datasets[split] = dataset_cls(
                vis_root=vis_root,
                ann_root=ann_root,
                split=split,
                task=task,
                subset=subset,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy=frm_sampling_strategy,
                num_frm=num_frm,
                frame_height=frame_height,
                frame_width=frame_width,
                fps=fps,
                k=k
            )

        return datasets

@registry.register_builder("downstream_tasks_qa_nextqa_224x224_5fps")
class DownstreamTasksQA_NextQA_Builder(BaseDatasetBuilder):

    train_dataset_cls = DownstreamTask_QA_NextQA
    eval_dataset_cls = DownstreamTask_QA_NextQA

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/downstream_tasks/downstream_tasks_qa_nextqa_224x224_5fps.yaml"
    }
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # TODO: add downloading function
        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        #  Overwrite to build() in BaseDatasetBuilder.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        fps = self.config.get('fps', None) # None for not using start and end time

        # additional arguments        
        train_k = self.config.get('train_k', None)
        eval_k = self.config.get('eval_k', None)
        print("## if take subset during training:", train_k)
        print("## if take subset during evaluation:", eval_k)
        
        frm_sampling_strategy = self.config.get('frm_sampling_strategy', 'uniform')
        num_frm = self.config.get('num_frm', 8)
        train_frame_height = self.config.get('train_frame_height', 224)
        train_frame_width = self.config.get('train_frame_width', 224)
        eval_frame_height = self.config.get('eval_frame_height', 224)
        eval_frame_width = self.config.get('eval_frame_width', 224)
        eval_only = self.config.get('eval_only', False)
        task = self.config.get('task', 'mc')
        
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            if eval_only and is_train:
                continue

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_root = ann_info.get(split).get('path')
            vis_root = build_info.get('videos').get('path')
            print("ann_root:", ann_root)
            print("vis_root:", vis_root)

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            # print('split:', split, '| ann_path:', ann_paths)
            if split == "train":
                frame_height = train_frame_height
                frame_width = train_frame_width
                k = train_k
            else:
                frame_height = eval_frame_height
                frame_width = eval_frame_width
                k = eval_k

            datasets[split] = dataset_cls(
                vis_root=vis_root,
                ann_root=ann_root,
                split=split,
                task=task,
                vis_processor=vis_processor,
                text_processor=text_processor,
                frm_sampling_strategy=frm_sampling_strategy,
                num_frm=num_frm,
                frame_height=frame_height,
                frame_width=frame_width,
                fps=fps,
                k=k
            )

        return datasets