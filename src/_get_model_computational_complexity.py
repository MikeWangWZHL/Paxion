import torchvision.models as models
import torch
import torch.nn as nn

from ptflops import get_model_complexity_info

import InternVideo
from models import (
    PatchAndFuseInternVideo,
    KnowledgePatcherInternVideo_Baseline_Simple
)
def set_up_device(gpu_index):
    # single gpu
    if torch.cuda.is_available() and gpu_index != -1:
        dev = f"cuda:{gpu_index}"
    else:
        dev = "cpu"
    return torch.device(dev)

class Wrapper(nn.Module):
    def __init__(self, module) -> None:
        super(Wrapper, self).__init__()
        self.net = module
    def forward(self, x):
        return self.net.encode_video(x)

with torch.cuda.device(1):
    # device = set_up_device(gpu_index=3)

    for model_name in [
        "patch_and_fuse_internvideo",
        "patch_and_fuse_internvideo_baseline_simple"
    ]:

        print("model_name:",model_name)

        model_type = "InternVideo-MM-L-14"
        print("model_type:",model_type)

        # load_model
        if model_name == "patch_and_fuse_internvideo":
            module = PatchAndFuseInternVideo.from_pretrained(model_type=model_type)
        elif model_name == "patch_and_fuse_internvideo_baseline_simple":
            module = KnowledgePatcherInternVideo_Baseline_Simple.from_pretrained(model_type=model_type)

        model = Wrapper(module)

        macs, params = get_model_complexity_info(model, (8,3,224,224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))