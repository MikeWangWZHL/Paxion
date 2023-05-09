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
    parser = argparse.ArgumentParser(description="Evaluate")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
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


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

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

    # build model
    model = task.build_model(cfg)

    # build runner

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    test_logs, val_logs = runner.evaluate(skip_reload=True)
    print("val logs:", val_logs)
    print("test logs:", test_logs)

    with open(os.path.join(runner.result_dir, "val_output.json"), 'w') as out:
        json.dump(val_logs, out, indent=4)

    with open(os.path.join(runner.result_dir, "test_output.json"), 'w') as out:
        json.dump(test_logs, out, indent=4)

if __name__ == "__main__":
    main()
