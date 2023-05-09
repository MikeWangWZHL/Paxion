import re
import torch
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from lavis.processors.blip_processors import BlipImageBaseProcessor


@registry.register_processor("vl_dynamic_blip_ego4d_image_train")
class VLDynamicModelBlipPretrainEgo4dImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.9, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop( # maintain most of the image since the action may only affect a small area
                    image_size,
                    scale=(min_scale, max_scale), 
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # transforms.RandomHorizontalFlip(), # left and right hand alignment
                RandomAugment(
                    2,
                    3, # lower level, original is 5
                    isPIL=True,
                    augs=[
                        "Identity",
                        # "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, video):
        if isinstance(video, list):
            video = [self.transform(item) for item in video]
            return torch.stack(video) # (num_frm, C, H, W)
        else:
            return self.transform(video)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.9) # 0.5
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

@registry.register_processor("vl_dynamic_blip_ego4d_image_eval")
class VLDynamicModelBlipPretrainEgo4dImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, video):
        if isinstance(video, list):
            video = [self.transform(item) for item in video]
            return torch.stack(video) # (num_frm, C, H, W)
        else:
            return self.transform(video)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)

@registry.register_processor("minimum_text")
class MinimumTextProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + caption
        return caption
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)
        return cls(prompt=prompt, max_words=max_words)

@registry.register_processor("vl_dynamic_ego4d_text")
class VLDynamicModelBlipPretrainEgo4dTextProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):

        # replace ego4d annotation tags
        caption = re.sub(r"\. #unsure", "", caption.lower())
        caption = re.sub(r"the #unsure", "something", caption)
        caption = re.sub(r"a #unsure", "something", caption)
        caption = re.sub(r" #unsure", " something", caption)
        caption = re.sub(r"#([a-z]) ", "", caption)
        caption = re.sub(r"# ([a-z]) ", "", caption)
        # caption = re.sub(r" woman ([a-z])", " a woman", caption)
        # caption = re.sub(r" man ([a-z])", " a man", caption)
        # caption = re.sub(r" person ([a-z])", " a person", caption)
        # caption = re.sub(r"^c ", "the person ", caption)
        # caption = re.sub(r"^([d-z]) ", "a person ", caption)

        # weird punctuations -> single space
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption,
        )
        # more than 2 white spaces -> single space
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


from InternVideo import video_transform
from einops import rearrange

@registry.register_processor("internvideo_eval")
class InternVideoVideoProcessorEval(BaseProcessor):
    def __init__(self, 
        image_size=224, 
        scale_size=224,
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    ):
        
        self.transform = transforms.Compose([
            # video_transform.TensorToNumpy(), # turn tensor to list of pil images
            video_transform.Resize(scale_size),
            video_transform.CenterCrop(image_size),
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=mean, std=std)
        ])
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, video):
        # item: a list of PIL Image with shape (num_frm, (H, W, C))
        if isinstance(video, list):
            video = self.transform(video)
        else:
            video = self.transform([video])
        video = rearrange(video, 'c m h w -> m c h w') # be consistent for all processors
        return video #(num_frm, C, H, W)
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        scale_size = cfg.get("scale_size", 224)
        mean = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
        std = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])

        return cls(
            image_size=image_size, 
            scale_size=scale_size,
            mean=mean,
            std=std
        )

@registry.register_processor("video_train")
class VideoProcessorTrain(BaseProcessor):
    def __init__(self, 
        image_size=224, 
        scale_size=224,
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711],
        min_scale = 0.9,
        max_scale = 1.0
    ):
        ## BLIP transform
        # self.transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop( # maintain most of the image since the action may only affect a small area
        #             image_size,
        #             scale=(min_scale, max_scale), 
        #             interpolation=InterpolationMode.BICUBIC,
        #         ),
        #         # transforms.RandomHorizontalFlip(), # left and right hand alignment
        #         RandomAugment(
        #             2,
        #             5,
        #             isPIL=True,
        #             augs=[
        #                 "Identity",
        #                 # "AutoContrast",
        #                 "Color",
        #                 "Brightness",
        #                 "Sharpness",
        #                 "Equalize",
        #                 "ShearX",
        #                 "ShearY",
        #                 "TranslateX",
        #                 "TranslateY",
        #                 "Rotate",
        #             ],
        #         ),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean, std)
        #     ]
        # )
    
        ## internvideo transform
        self.transform = transforms.Compose([
            # video_transform.TensorToNumpy(), # turn tensor to list of pil images
            video_transform.Resize(scale_size),
            video_transform.CenterCrop(image_size),
            # video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),  # Randomly change the brightness, contrast and saturation and hue of the clip
            video_transform.ColorJitter(0.1, 0.1, 0.1, 0.1),  # Randomly change the brightness, contrast and saturation and hue of the clip
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=mean, std=std)
        ])

        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, video):
        # video: a list of PIL Image with shape (num_frm, (H, W, C))

        ## BLIP transform
        # if isinstance(video, list):
        #     video = [self.transform(item) for item in video]
        #     return torch.stack(video) # (num_frm, C, H, W)
        # else:
        #     return self.transform(video)

        ## intern video transform
        if isinstance(video, list):
            video = self.transform(video)
        else:
            video = self.transform([video])
        video = rearrange(video, 'c m h w -> m c h w') # be consistent for all processors
        return video


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        # scale_size = cfg.get("scale_size", 256)
        scale_size = cfg.get("scale_size", 224) # cover the entire frame, since ego4d can have actions at the very edge of the frames
        mean = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
        std = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])

        return cls(
            image_size=image_size, 
            scale_size=scale_size,
            mean=mean,
            std=std
        )
    