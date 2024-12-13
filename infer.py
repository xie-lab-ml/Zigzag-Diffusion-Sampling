import torch
import os
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args for Z-sampling")

    parser.add_argument('--gamma_1', type=float, default=5.5, help='guidance for denoising process')
    parser.add_argument('--gamma_2', type=float, default=0, help='guidance for inversion process')
    parser.add_argument('--lambda_step', type=int, default=49, help='zigzag timestep')
    parser.add_argument('--infer_step', type=int, default=50, help='total inference timestep T')
    parser.add_argument('--image_size', type=int, default=1024, help='The size (height and width) of the generated image.')
    parser.add_argument('--T_max', type=int, default=1, help='Number of rounds for each zigzag iteration step.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to determine the initial latent.')
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--save_dir', type=str, default='./res', help='Path to save the generated images.')

    args = parser.parse_args()
    return args

#TODO user could change this list
prompt_list = ['A Man on a Bicycle, MineScaft Style', 'A small yellow dog.',"A Man on a Bicycle, in the style of Van Gogh's Starry Night."]

def get_init_latents(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)
    start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
    return start_latents

if __name__ == '__main__':
    #load args
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    assert args.lambda_step < args.infer_step

    #load model
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                     variant='fp16',
                                                     safety_checker=None, requires_safety_checker=False)
    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                             subfolder='scheduler')

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inv_scheduler = inverse_scheduler
    device = torch.device(args.device)
    pipe = pipe.to(args.device)


    shape = (1, 4, args.image_size // 8, args.image_size // 8)
    #initial latent x_{T}
    init_latent = get_init_latents(args.seed)

    for idx, prompt in enumerate(prompt_list):
        print(f'idx: {idx}\tprompt: {prompt}')
        #use standard sampling to generate image
        print('start generation via Standard-Sampling...')
        origin_img = pipe(prompt=prompt, shape=shape, guidance_scale = args.gamma_1,
                                num_inference_steps=args.infer_step,latents=init_latent).images[0]
        origin_img.save(os.path.join(args.save_dir,f'{idx}_origin_image_{prompt}.png'))

        #use z-sampling to gnerate image
        print('start generation via Z-Sampling...')
        z_sampling_img = pipe.z_sampling_call(prompt=prompt, shape=shape,
                                          guidance_scale=args.gamma_1, inv_guidance_scale=args.gamma_2, num_inference_steps=args.infer_step,
                                          latents=init_latent, T_max=args.T_max, lambda_step=args.lambda_step).images[0]
        z_sampling_img.save(os.path.join(args.save_dir,f'{idx}_z_sampling_image_{prompt}.png'))

    print('The End!')