"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
import os
from typing import Literal
import time
from tqdm import tqdm

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video

class T2VModel():
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """
    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        lora_rank: int = 128,
        image_or_video_path: str = "",
        generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
        dtype: torch.dtype = torch.bfloat16):
        # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
        # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
        # function to use Multi GPUs.

        self.image = None
        self.video = None

        if generate_type == "i2v":
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
            self.image = load_image(image=image_or_video_path)
        elif generate_type == "t2v":
            self.pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        else:
            self.pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
            self.video = load_video(image_or_video_path)

        # If you're using with lora, add this code
        if lora_path:
            self.pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
            self.pipe.fuse_lora(lora_scale=1 / lora_rank)

        # 2. Set Scheduler.
        # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
        # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
        # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

        # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

        # 3. Enable CPU offload for the model.
        # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
        # and enable to("cuda")

        # pipe.to("cuda")

        self.pipe.enable_sequential_cpu_offload()

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_video(
        self,
        prompt: str,
        output_path: str = "./output.mp4",
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
        seed: int = 42,
    ):
        # 4. Generate the video frames based on the prompt.
        # `num_frames` is the Number of frames to generate.
        # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
        if generate_type == "i2v":
            video_generate = self.pipe(
                prompt=prompt,
                image=self.image,  # The path of the image to be used as the background of the video
                num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
                num_inference_steps=num_inference_steps,  # Number of inference steps
                num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
                use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            ).frames[0]
        elif generate_type == "t2v":
            video_generate = self.pipe(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=5,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
        else:
            video_generate = self.pipe(
                prompt=prompt,
                video=self.video,  # The path of the video to be used as the background of the video
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                # num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            ).frames[0]
        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        export_to_video(video_generate, output_path, fps=8)

def check_prompt_args(prompt, prompt_file):
    if prompt is None and prompt_file is None:
        raise ValueError("Either 'prompt' or 'prompt_file' must be provided")
    if prompt is not None and prompt_file is not None:
        raise ValueError("Only one of 'prompt' or 'prompt_file' can be provided")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="The description of the video to be generated")
    parser.add_argument("--prompt_file", type=str, help="Prompt file")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./results/output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    check_prompt_args(args.prompt, args.prompt_file)
    
    print("Init model")
    start = time.time()
    model = T2VModel(
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        image_or_video_path=args.image_or_video_path,
        generate_type=args.generate_type,
        dtype=dtype,
    )
    print(f"Model init time: {time.time() - start:.2f}s")

    print("Generate video")
    start = time.time()
    if args.prompt is not None:
        print(f"Prompt: {args.prompt}, len: {len(args.prompt)}")
        model.generate_video(
            prompt=args.prompt,
            output_path=args.output_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos_per_prompt,
            generate_type=args.generate_type,
            seed=args.seed,
        )
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            prompts = f.read().splitlines()
            for prompt in tqdm(prompts):
                # Remove the quotes from the prompt
                prompt = prompt.strip('"')
                print(f"Prompt: {prompt}, len: {len(prompt)}")
                model.generate_video(
                    prompt=prompt,
                    output_path="./results/" + prompt.replace(" ", "_") + ".mp4",
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    generate_type=args.generate_type,
                    seed=args.seed,
                )
    else:
        raise ValueError("Either 'prompt' or 'prompt_file' must be provided")
    print(f"Video generation time: {time.time() - start:.2f}s")
    print("Done")
