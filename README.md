# Repo for paper: "Paxion: Patching Action Knowledge in Video-Language Foundation Models"

### A note regarding the naming convention in this repo: we use "PatchAndFuse" or "patch_and_fuse" as an alternative name for "Paxion".

# Setup
## Environment Setup 
- after cloning this repo, setup submodules by:
    ```
    git submodule update --init --recursive
    ```
    <!-- Side note: some modifications on the initial environment.yml file: 
    - made the pip install part more compact with `-r requirements.txt`
    - removed `en-core-web-sm==3.4.1` because currently it cannot be installed with pip
    - removed torch related lines because it seems better to install using a separate command -->
- Setup conda environment
    ```
    conda env create -f environment.yml
    conda activate paxion
    ```
    <!-- pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 -->

- install LAVIS library from source
    ```
    cd src/LAVIS
    pip install -e .
    ```

## Dataset Setup
### Download Annotations
- Download the annotations for **ActionBench** [here]() (Link removed for anonymity); and put under `ActionBench/ego4d` and `ActionBench/ssv2`
- Download the annotations for **downstream tasks** [here]() (Link removed for anonymity); and put the downloaded folder under the root directory of this repo as `datasets/`
- **Annotation details** for each dataset can be found in the `.md` files in [dataset_cards](./dataset_cards/).

### Download Videos & Preprocessing
Please refer to the `.md` files in [dataset_cards](./dataset_cards/) for instructions on downloading the raw videos and preprocessing.

## Download Pretrained Backbone Checkpoints
- **InternVideo**: Download the `InternVideo-MM-L-14` checkpoint following the instructions [here](https://github.com/OpenGVLab/InternVideo#model-zoo); put the downloaded `InternVideo-MM-L-14.ckpt` under `src/pretrained_ckpt/InternVideo/InternVideo-MM-L-14.ckpt`
- **ClipViP**: Download the `CLIP-ViP-B/32` checkpoint following the instructions [here](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP#getting-started); put the downloaded `pretrain_clipvip_base_32.pt` under `src/pretrained_ckpt/ClipViP/pretrain_clipvip_base_32.pt`
- **Singularity**: Download the `Pre-trained checkpoints` following the instructions [here](https://github.com/jayleicn/singularity#download); put the `singularity_temporal_17m` under `src/pretrained_ckpt/Singularity/singularity_temporal_17m.pth`

## Download Trained Knowledge Patcher and Knowledge Fuser
- Download the Knowledge Patcher checkpoints on actionbench [here]() (Link removed for anonymity); and put under `src/pretrained_ckpt/PatchAndFuse/ActionBench`
- Download the Patch & Fuse checkpoints on downstream tasks [here]() (Link removed for anonymity); and put under `src/pretrained_ckpt/PatchAndFuse/downstream_tasks`

# Quick Start
**[demo.py](./src/demo.py)** shows an usage example on loading a trained PatchAndFuse model (with InternVideo backbone and trained on SSv2-label) and perform inference on video-text matching. 

# Code Description
We build our codebase on top of [LAVIS framework](https://github.com/salesforce/LAVIS/tree/main). Please refer to the [documentation](https://opensource.salesforce.com/LAVIS//latest/intro.html#library-design) to get an idea of the overall structure (Tasks, Models, Runners, etc). Config files for running any tasks can be found in `src/configs/projects`, we include detailed comments in the `.yaml` files for more fine-grained customization on the experimental configs.

## Training
To train Knowledge Patcher on ActionBench or further train Knowledge Fuser on downstream tasks, we provide configs under `src/configs/projects/train`; Please make sure to look into the configs and do necessary modifications (e.g. specify trained checkpoints which are marked as `#TODO`). 

Here is an example for using the training configs (`run_scripts/train.sh`):
    ```
        cd src/
        bash run_scripts/train.sh
    ```

## Evaluation
To evaluate Knowledge Patcher on ActionBench or evaluate trained Knowledge Fuser on downstream tasks, we provide configs under `src/configs/projects/eval`; Please make sure to look into the configs and do necessary modifications (e.g. specify trained checkpoints which are marked as `#TODO`). 

Here are two examples for using the evaluation configs (`run_scripts/eval_actionbench.sh` and `run_scripts/eval_downstream_task.sh` ):
- eval actionbench:
    ```
        cd src/
        bash run_scripts/eval_actionbench.sh 
    ```
- eval downstream task:
    ```
        cd src/
        bash run_scripts/eval_downstream_task.sh 
    ```

## Inference
For customized inference, i.e., evaluate on some specific samples and visualize the results, we provide [inference.py](./src/inference.py) where one can set spefic instance ids in the `__main__` function. 

Examples for running inference on different tasks can be found in `run_scripts/inference.sh`

# Acknowledgement
This code used resources from [LAVIS](https://github.com/salesforce/LAVIS), [InternVideo](https://github.com/OpenGVLab/InternVideo), [ClipViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP), [Singularity](https://github.com/jayleicn/singularity), and [flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch). The code is implemented using PyTorch. We thank the authors for their great work and open-sourcing the code. 

