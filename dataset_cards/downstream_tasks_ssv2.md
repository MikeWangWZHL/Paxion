## Instruction for Downloading Videos
refer to [acdybench_ssv2.md](./acdybench_ssv2.md)

## Downstream Task: SSv2-label and SSv2-template
- paper: https://arxiv.org/abs/2206.03428
- train size: 168913
- val size: 2088
- ann_path: `/shared/nas/data/m1/wangz3/Paper_Code_Repos_Cleaned/neurips23_patch_and_fuse/datasets/SSv2/ssv2_label_ssv2_template`
- format SSv2-label:
    ```
        [
            {"video": ["62211.webm"], "caption": "spinning soap that quickly stops spinning"},
        ]
    ```
- format SSv2-template:
    ```
        [
            {"video": ["62211.webm", "63095.webm", "174825.webm", "65027.webm", "65677.webm", "37955.webm", "9741.webm", "47588.webm", "31811.webm", "155308.webm", "6483.webm", "106444.webm"], "caption": "Spinning [something] that quickly stops spinning"}
        ]
    ```