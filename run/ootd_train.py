import os

import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
import argparse
from utils.dataloader import VITONDataset,VITONDataLoader,make_train_dataset
from safetensors.torch import save_file

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='/mmu-vcg-ssd/yichen/VTONDataset/VITON-HD/zalando-hd-resized/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')


    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./OOTD-model",
        help="The output directory where the model predictions and checkpoints will be written.",)
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--model_type", type=str, default='hd', help="hd or dc."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(

        "--conditioning_dropout_prob",
        type = float,
        default=0.1,
        
    )
    
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    return opt


import sys
# sys.argv = ['ootd_train.py']
opt = get_opt()
opt.batch_size = opt.train_batch_size

test_dataset = VITONDataset(opt)
test_loader = VITONDataLoader(opt, test_dataset)
# test_loader.data_loader.sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
# torch.distributed.init_process_group(backend="nccl", rank=0, world_size=2)
train_dataset = test_dataset
train_dataloader = test_loader.data_loader

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler,PNDMScheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
sys.path.append(r'/mmu-vcg-ssd/yichen/OOTDiffusion/ootd')

from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel

VIT_PATH = "/mmu-vcg-ssd/yichen/OOTDiffusion/checkpoints/clip-vit-large-patch14"
VAE_PATH = "/mmu-vcg-ssd/yichen/OOTDiffusion/checkpoints/ootd"
UNET_PATH = "/mmu-vcg-ssd/yichen/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/"
MODEL_PATH = "/mmu-vcg-ssd/yichen/OOTDiffusion/checkpoints/ootd"
scheduler_path = '/mmu-vcg-ssd/yichen/OOTDiffusion/checkpoints/ootd/scheduler/scheduler_ootd_config.json'

vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm_train",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton_train",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
################
if unet_vton.conv_in.in_channels == 4:
    print("sxxxxxxxxxx")
    with torch.no_grad():
        new_in_channels = 8
        # Replace the first conv layer of the unet with a new one with the correct number of input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet_vton.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        
        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer
        
        conv_new.weight.data[:, :4] = unet_vton.conv_in.weight.data  # Copy weights from old conv layer
        conv_new.bias.data = unet_vton.conv_in.bias.data  # Copy bias from old conv layer
        
        unet_vton.conv_in = conv_new  # replace conv layer in unet
        print('#######Replace the first conv layer of the unet with a new one with the correct number of input channels#######')
        # unet_garm.config['in_channels'] = new_in_channels  # update config
#################


noise_scheduler = PNDMScheduler.from_config(scheduler_path)
        
auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)

tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        )
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
# register_to_config(requires_safety_checker=requires_safety_checker)

vae.requires_grad_(False)
unet_garm.requires_grad_(True)
unet_vton.requires_grad_(True)
image_encoder.requires_grad_(False)
text_encoder.requires_grad_(False)

unet_garm.train()
unet_vton.train()
 # Optimizer creation
import math
from pathlib import Path
args = opt
logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    
    )

optimizer_class = torch.optim.AdamW
######单机单卡
params_to_optimize = list(unet_garm.parameters()) + list(unet_vton.parameters())
optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
#######单机多卡
# params_to_optimize = list(unet_garm.parameters())
# optimizer = optimizer_class(
#         params_to_optimize,
#         lr=args.learning_rate,
#         betas=(args.adam_beta1, args.adam_beta2),
#         weight_decay=args.adam_weight_decay,
#         eps=args.adam_epsilon,
#     )
# optimizer.add_param_group({'params': list(unet_vton.parameters())})

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
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

# if accelerator.state.deepspeed_plugin is not None:
#   kwargs = {
              
#               "train_micro_batch_size_per_gpu": 1,
#               "train_batch_size": 1,
              
#           } 
#     accelerator.state.deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)c


#Prepare everything with our `accelerator`.
class Unet_(torch.nn.Module):
    def __init__(self, vton_model, garm_model):
        super(Unet_, self).__init__()
        self.unet_vton = vton_model
        self.unet_garm = garm_model
        # 其他初始化代码...

unet_ = Unet_(unet_vton, unet_garm)

unet_ ,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet_, optimizer, train_dataloader, lr_scheduler
    )



# unet_garm,unet_vton,optimizer, train_dataloader,test_loader, lr_scheduler = accelerator.prepare(
#          unet_garm,unet_vton,optimizer, train_dataloader, test_loader,lr_scheduler
#     )

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move vae, unet and text_encoder to device and cast to weight_dtype
vae.to(accelerator.device, dtype=weight_dtype)
unet_garm.to(accelerator.device)
unet_vton.to(accelerator.device)
image_encoder.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)


# unet_garm.to(dtype=weight_dtype)
# unet_vton.to(dtype=weight_dtype)




# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

def tokenize_captions( captions, max_length):
        inputs = tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

batchsize = args.train_batch_size

from accelerate.logging import get_logger
from tqdm import tqdm
logger = get_logger(__name__)
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

initial_global_step = 0

# progress_bar = tqdm(
#         range(0, args.max_train_steps),
#         initial=initial_global_step,
#         desc="Steps",
#         # Only show the progress bar once on each machine.
#         disable=not accelerator.is_local_main_process,
#     )

image_logs = None
for epoch in tqdm(range(first_epoch, args.num_train_epochs)):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet_vton),accelerator.accumulate(unet_garm):
        # with accelerator.accumulate(unet_vton):



            image_garm = batch['cloth']['paired'].to(accelerator.device).to(dtype=weight_dtype)
            image_vton = batch['img_agnostic'].to(accelerator.device).to(dtype=weight_dtype)
            image_ori = batch['img'].to(accelerator.device).to(dtype=weight_dtype)
            
            #get prompt embeds
            prompt_image = auto_processor(images=image_garm, return_tensors="pt").to(accelerator.device)
            prompt_image = image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if args.model_type == 'hd':

                prompt_embeds = text_encoder(tokenize_captions(['']*batchsize, 2).to(accelerator.device))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif args.model_type == 'dc':
                prompt_embeds = text_encoder(tokenize_captions([category], 3).to(accelerator.device))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")
                
            ######preprocess把[0,1]转为【-1，1】

            image_garm = image_processor.preprocess(image_garm)
            image_vton = image_processor.preprocess(image_vton)
            image_ori = image_processor.preprocess(image_ori)
                
            # Convert images to latent space

            latents = vae.encode(image_ori).latent_dist.sample()
            # latents = vae.encode(image_ori.to(weight_dtype).latent_dist.sample().to(accelerator.device))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 2. Encode input prompt
            prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            num_images_per_prompt = 1
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # 3. Preprocess image
            # image_garm = batch['cloth']
            # image_vton = batch['agnostic']
            # image_ori = batch['img']
            # image_garm = image_processor.preprocess(image_garm)
            # image_vton = image_processor.preprocess(image_vton)
            # image_ori = image_processor.preprocess(image_ori)
            # mask = np.array(mask)
            # mask[mask < 127] = 0
            # mask[mask >= 127] = 255
            # mask = torch.tensor(mask)
            # mask = mask / 255
            # mask = mask.reshape(-1, 1, mask.size(-2), mask.size(-1))

            # 4. set timesteps ,在生成noisy_latents之前设置过了
            
            # 5. Prepare Image latents
            # garm_latents = self.prepare_garm_latents(
            #     image_garm,
            #     batch_size,
            #     num_images_per_prompt,
            #     prompt_embeds.dtype,
            #     device,
            #     self.do_classifier_free_guidance,
            #     generator,
            # )
            image_latents_garm = vae.encode(image_garm).latent_dist.mode()
            image_latents_garm = torch.cat([image_latents_garm], dim=0)
            
            # vton_latents, mask_latents, image_ori_latents = self.prepare_vton_latents(
            #     image_vton,
            #     mask,
            #     image_ori,
            #     batch_size,
            #     num_images_per_prompt,
            #     prompt_embeds.dtype,
            #     device,
            #     self.do_classifier_free_guidance,
            #     generator,
            # )
            image_latents_vton = vae.encode(image_vton).latent_dist.mode()
            # image_ori_latents = vae.encode(image_ori).latent_dist.mode()
            
            image_latents_vton = torch.cat([image_latents_vton], dim=0)
            # image_ori_latents = torch.cat([image_ori_latents], dim=0)

            # height, width = image_latents_vton.shape[-2:]
            # height = height * vae_scale_factor
            # width = width * vae_scale_factor
            #####  modify: we should dropout the cloth condition! ##########      
            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(bsz, device=latents.device)
                #########################################################

                # Sample masks for the cloth images.
                image_mask_dtype = image_latents_garm.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(bsz, 1, 1, 1)
                # Final image conditioning.
                image_latents_garm = image_mask * image_latents_garm
            ####################################################################

            sample,spatial_attn_outputs = unet_garm(
            image_latents_garm,
            0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,)


            # unet_garm(image_latents_garm,0,encoder_hidden_states=prompt_embeds,return_dict=False,)
            # import pdb;pdb.set_trace()

            latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)

            spatial_attn_inputs = spatial_attn_outputs.copy()
            
            noise_pred = unet_vton(
                    latent_vton_model_input,
                    spatial_attn_inputs,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]


            # with accelerator.autocast():
            util_adv_loss = torch.nn.functional.softplus(-sample).mean() * 0 
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")+util_adv_loss
            
            print(loss.item())

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # progress_bar.update(7)#####单机多卡，每次增加卡的数量
            accelerator.log({"training_loss": loss}, step=step)
    if (epoch%100 == 0 and epoch != 0 ) or epoch == (args.num_train_epochs-1):
        state_dict_unet_vton = unet_vton.state_dict()
        for key in state_dict_unet_vton.keys():
            state_dict_unet_vton[key] = state_dict_unet_vton[key].to('cpu')
        save_file(state_dict_unet_vton, f"./ootd_train_checkpoints/unet_vton-epoch{str(epoch)}.safetensors")
        # save_file(state_dict_unet_vton, f"./ootd_train_checkpoints/unet_vton.safetensors")
        state_dict_unet_garm = unet_garm.state_dict()
        for key in state_dict_unet_garm.keys():
            state_dict_unet_garm[key] = state_dict_unet_garm[key].to('cpu')
        save_file(state_dict_unet_garm,f"./ootd_train_checkpoints/unet_garm-epoch{str(epoch)}.safetensors")
        # save_file(state_dict_unet_garm,f"./ootd_train_checkpoints/unet_garm.safetensors")                
        print('checkpoints successful saved')
accelerator.end_training()
# from safetensors.torch import save_file
# save_file(unet_vton.to('cpu').state_dict(), "./unet_vton.safetensors")
# save_file(unet_garm.to('cpu').state_dict(),"./unet_garm.safetensors")