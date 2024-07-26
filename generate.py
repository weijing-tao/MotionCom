import torch
import argparse
from random import sample
from motionpaint import MotionComDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os
from PIL import Image

def process_video(args):
    # Load and prepare the image and mask
    image = load_image(args.image_path).resize((1024, 576))
    mask_img = load_image(args.mask_path).resize((1024, 576))

    # Load the pipeline
    pipe = MotionComDiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Generate unique seeds if needed
    if args.num_videos > 1:
        seed_range = 1000000  # Define the upper limit for seed range
        seeds = sample(range(seed_range), args.num_videos)
    else:
        seeds = [args.seed]  # Use the specified single seed

    # Generate and process videos
    for seed in seeds:
        generator = torch.manual_seed(seed)
        output_path = f"{args.output_base_path}{seed}.mp4"
        print('Processing seed:', output_path)

        # Run your pipeline with the specified parameters
        frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=args.motion_bucket_id, noise_aug_strength=args.noise_aug_strength, overlay_init_image=image, overlay_mask_image=mask_img).frames[0]
        
        # Optionally save each frame as an image
        if args.save_frames:
            frame_folder = f"{args.output_base_path}frames_seed{seed}/"
            os.makedirs(frame_folder, exist_ok=True)
            for idx, img in enumerate(frames):
               
                img.save(f"{frame_folder}frame_{idx}.png")
            print(f"All frames saved in: {frame_folder}")
        # Export the frames to a video file
        export_to_video(frames, output_path, fps=args.fps)

        # Print out the path to the saved video
        print(f"Video saved in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process videos with customized motion composition.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask image')
    parser.add_argument('--output_base_path', type=str, required=True, help='Base path for output videos')
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-video-diffusion-img2vid-xt', help='Path to the pretrained model')
    parser.add_argument('--motion_bucket_id', type=int, default=60, help='Motion bucket ID for the diffusion model')
    parser.add_argument('--num_videos', type=int, default=1, help='Number of videos to generate')
    parser.add_argument('--seed', type=int, default=None, help='Seed for generating the video')
    parser.add_argument('--noise_aug_strength', type=float, default=0.02, help='Noise augmentation strength')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second in the output video')
    parser.add_argument('--save_frames', action='store_true', help='Save each frame as an image before creating video')

    args = parser.parse_args()
    
    # Set default seed if not provided and num_videos is 1
    if args.seed is None and args.num_videos == 1:
        args.seed = 42  # Default seed value

    # Check if both seed and num_videos are provided and num_videos > 1
    if args.seed is not None and args.num_videos > 1:
        parser.error("Error: --seed and --num_videos cannot be used together when num_videos > 1. If a seed is specified, only one video can be generated to ensure determinism.")
    
    # Extract the base directory from the output path
    base_dir = os.path.dirname(args.output_base_path)
    
    # Check if the base directory exists, create it if it doesn't
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")

    process_video(args)

if __name__ == '__main__':
    main()
