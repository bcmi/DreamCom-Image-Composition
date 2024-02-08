import os
import argparse
import hashlib
import itertools
import math
import torch
import random
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import albumentations as A
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
import cv2


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def prepare_mask_and_masked_image(image, mask,loss_mask):

    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    loss_mask = np.array(loss_mask.convert("L"))
    loss_mask  = loss_mask .astype(np.float32) / 255.0
    loss_mask  = loss_mask [None, None]
    loss_mask [loss_mask  < 0.5] = 0
    loss_mask [loss_mask  >= 0.5] = 1
    loss_mask = torch.from_numpy(loss_mask)

    return mask, masked_image,loss_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--background_prompt",
        type=str,
        default='background',
        help="The prompt for background.",
    )
    parser.add_argument(
        "--num_inside",
        type=int,
        default=38,
        help="num inside",
    )
    parser.add_argument(
        "--num_outside",
        type=int,
        default=38,
        help="num outside",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--image_num",
        type=int,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--output_medium", action="store_true", help="Whether or not to output meidum result."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--package_name",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    parser.add_argument(
        "--fg_name",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            background_prompt='background',
            size=512,
            center_crop=False,
            image_num=5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)

        if not self.instance_data_root.exists():
            print(self.instance_data_root)
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = []
        self.instance_masks_path = []

        count=0
        for temp_file_path in Path(instance_data_root).glob("*.jpg"):
            self.instance_images_path.append(temp_file_path)
            self.instance_masks_path.append(Path(str(temp_file_path).replace('jpg', 'png')))
            count+=1
            if count>=image_num:
                break


        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.background_prompt=background_prompt
        self._length1 = self.num_instance_images
        self._length2 = 0

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)

            with open(class_data_root + '/bbox.txt', 'r') as file:
                lines = file.readlines()
            class_boxes = []
            for line in lines:
                box = line.strip().split(',')
                class_boxes.append(
                    [round(float(box[0])), round(float(box[1])), round(float(box[2])), round(float(box[3]))])

            self.class_images_path = []
            self.class_boxes = []

            for temp_file_path in Path(class_data_root).glob("*.jpg"):
                temp_name = str(temp_file_path).replace(class_data_root + '/', '').replace('.jpg', '')
                self.class_images_path.append(temp_file_path)
                self.class_boxes.append(class_boxes[int(temp_name)])

            self.num_class_images = len(self.class_images_path)
            print(self.num_class_images)
            self._length2 =self.num_class_images
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_transforms_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
        self.image_transforms_crop = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=size, height=size),
                A.ShiftScaleRotate(scale_limit=(-0.2,0.2),shift_limit=0,rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=2,p=0.6),
                
            ], is_check_shapes=False
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # self.mask_transforms=transforms.ToTensor()

    def __len__(self):
        if self._length2 > 0:
            return 2 * self._length2
        else:
            return self._length1

    def mask_bboxregion_coordinate(self, mask):
        w, h = np.shape(mask)[:2]
        valid_index = np.argwhere(mask == 255)  # [length,2]
        if np.shape(valid_index)[0] < 1:
            x_left = 0
            x_right = 0
            y_bottom = 0
            y_top = 0
        else:
            x_left = np.min(valid_index[:, 0])
            x_right = np.max(valid_index[:, 0])
            y_bottom = np.min(valid_index[:, 1])
            y_top = np.max(valid_index[:, 1])

        return x_left, x_right, y_bottom, y_top

    def __getitem__(self, index):
        example = {}

        if self._length2 == 0 or index%2==0:
            if self._length2!=0:
                new_index=int(index/2)
            else:
                new_index=index
            instance_image = Image.open(self.instance_images_path[new_index % self.num_instance_images])

            # print(self.instance_masks_path[new_index % self.num_instance_images])
            # print(self.instance_images_path[new_index % self.num_instance_images])

            instance_mask = Image.open(self.instance_masks_path[new_index % self.num_instance_images]).convert('L')


            # mask膨胀处理
            mask = np.asarray(instance_mask) 
            m = np.array(mask>0).astype(np.uint8) 
            #m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
            instance_mask = Image.fromarray(m * 255) 


            x_left, x_right, y_bottom, y_top = self.mask_bboxregion_coordinate(np.array(instance_mask))
            W, H = (np.array(instance_mask)).shape[:2]
            mask_array_new = np.zeros((W,H))
            x_right=min(x_right,W-1)
            y_top=min(y_top,H-1)
            new_temp = np.ones((x_right - x_left + 1, y_top - y_bottom + 1)) * 255
            mask_array_new[x_left:x_right + 1, y_bottom:y_top + 1] = new_temp
            mask_array_new = np.uint8(mask_array_new)
            instance_mask = Image.fromarray(mask_array_new)


            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            #print(instance_image.size)
            
            instance_image = self.image_transforms_resize(instance_image)  # resize + random crop

            instance_mask = self.mask_transforms_resize(instance_mask)

            
            flag=True
            count_before=np.count_nonzero((np.array(instance_mask))>0)

            count_times=0

            compare_num=0.8

            while flag:
                transed = self.image_transforms_crop(image=np.array(instance_image), mask=np.array(instance_mask))
                mask = transed['mask'].copy()
                mask[mask == 2] = 0
                count_after = np.count_nonzero(mask >0)
             
                if count_after>compare_num*count_before:         
                    flag=False


                count_times+=1

                if count_times>=20:
                    compare_num-=0.02
                    count_times=0
                    print(compare_num)

            instance_image = Image.fromarray(transed['image'])
            instance_mask = Image.fromarray(mask)
            loss_mask=transed['mask']
            loss_mask[loss_mask!=2]=255
            loss_mask[loss_mask==2]=0
 

            instance_mask_loss = Image.fromarray(loss_mask)


            example["PIL_images"] = instance_image
            example["images"] = self.image_transforms(instance_image)  # to tensor + *2-1
            example["masks"] = instance_mask
            example["loss_masks"] = instance_mask_loss

            example["prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

            example["background_prompt_ids"] = self.tokenizer(
                self.background_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

            example["empty_prompt_ids"] = self.tokenizer(
                "",
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

        else:
            new_index = int(index / 2)

            class_image = Image.open(self.class_images_path[new_index % self.num_class_images])

            x_left, y_bottom, x_right, y_top = self.class_boxes[new_index % self.num_class_images]


            W, H = (np.array(class_image)).shape[:2]

            mask_array_new = np.zeros((W, H))

            y_top = min(y_top, W - 1)
            x_right = min(x_right, H - 1)

            new_temp = np.ones((y_top - y_bottom + 1, x_right - x_left + 1)) * 255
            mask_array_new[y_bottom:y_top + 1, x_left:x_right + 1] = new_temp
            mask_array_new = np.uint8(mask_array_new)
            class_mask = Image.fromarray(mask_array_new)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            class_image = self.image_transforms_resize(class_image)  # resize + random crop
            class_mask = self.mask_transforms_resize(class_mask)


            transed = self.image_transforms_crop(image=np.array(class_image), mask=np.array(class_mask))
            class_image = Image.fromarray(transed['image'])
            mask = transed['mask'].copy()
            mask[mask == 2] = 0
            class_mask = Image.fromarray(mask)
            loss_mask = transed['mask']
            loss_mask[loss_mask != 2] = 255
            loss_mask[loss_mask == 2] = 0
            class_mask_loss = Image.fromarray(loss_mask)



            example["PIL_images"] = class_image
            example["images"] = self.image_transforms(class_image)  # to tensor + *2-1
            example["masks"] = class_mask
            example["loss_masks"] = class_mask_loss

            example["prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

            example["background_prompt_ids"] = self.tokenizer(
                self.background_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

            example["empty_prompt_ids"] = self.tokenizer(
                "",
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_overflowing_tokens=True,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main():
    args = parse_args()
    args.instance_data_dir=args.instance_data_dir+args.package_name+'/'+args.fg_name
    args.instance_prompt=args.instance_prompt+args.class_name
    args.output_dir=args.output_dir+args.fg_name+str(args.image_num)
    print('training',args.class_name)


    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    accelerator.print(f'device {str(accelerator.device)} is used!')

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        print('class images: ',cur_class_images)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path+"/tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path+"/text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path+"/vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path+"/unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        background_prompt=args.background_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        image_num=args.image_num
    )

    def collate_fn(examples):
        input_ids = [example["prompt_ids"] for example in examples]
        background_input_ids = [example["background_prompt_ids"] for example in examples]
        empty_input_ids = [example["empty_prompt_ids"] for example in examples]

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding="max_length",max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        background_input_ids = tokenizer.pad({"input_ids": background_input_ids}, padding="max_length",max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        empty_input_ids = tokenizer.pad({"input_ids": empty_input_ids}, padding="max_length",max_length=tokenizer.model_max_length, return_tensors="pt").input_ids

        pixel_values = [example["images"] for example in examples]

        masks = []
        masked_images = []
        loss_masks=[]

        for example in examples:
            mask, masked_image,loss_mask = prepare_mask_and_masked_image(example["PIL_images"], example["masks"],example["loss_masks"])
            masks.append(mask)
            masked_images.append(masked_image)
            loss_masks.append(loss_mask)


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        loss_masks = torch.stack(loss_masks)

        batch = {"input_ids": input_ids, "background_input_ids": background_input_ids,"empty_input_ids": empty_input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images,"loss_masks":loss_masks}

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8).to(dtype=weight_dtype)

                loss_masks = batch["loss_masks"]
                loss_mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(new_mask, size=(args.resolution // 8, args.resolution // 8))
                        for new_mask in loss_masks
                    ]
                )
                loss_mask = loss_mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning\
                # import torch.autograd as autograd

                # autograd.set_detect_anomaly(True)

                prompt_inside_indices = torch.tensor([i for i in range(1, args.num_inside + 1)])
                prompt_outside_indices = torch.tensor([i for i in range(args.num_inside + 1, args.num_inside + args.num_outside + 1)])
                prompt_inside_collect = torch.tensor([i for i in range(1, args.num_inside + 1)])
                prompt_outside_collect = torch.tensor([i for i in range(1, args.num_outside + 1)])
                embedding_inside = text_encoder(batch["input_ids"])[0]
                embedding_outside = text_encoder(batch["background_input_ids"])[0]
                embedding_unconditional= text_encoder(batch["empty_input_ids"])[0]
                encoder_hidden_states = embedding_unconditional.clone()
                encoder_hidden_states[:, prompt_inside_indices, :] = embedding_inside[:, prompt_inside_collect, :]
                encoder_hidden_states[:, prompt_outside_indices, :] = embedding_outside[:, prompt_outside_collect, :]



                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states,inside_outside_mask=mask.clone()).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                #loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")


                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = ((loss * loss_mask).sum([1, 2, 3]) / loss_mask.sum([1, 2, 3])).mean()


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
