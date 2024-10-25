"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with quantization.

Note:

Must install the `torchao`，`torch`,`diffusers`,`accelerate` library FROM SOURCE to use the quantization feature.
Only NVIDIA GPUs like H100 or higher are supported om FP-8 quantization.

ALL quantization schemes must use with NVIDIA GPUs.

# Run the script:

python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b --quantization_scheme fp8 --dtype float16
python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --quantization_scheme fp8 --dtype bfloat16

"""

import argparse
import os
import time
from tqdm import tqdm

import torch
import torch._dynamo
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
# from torchao.float8.inference import ActivationCasting, QuantConfig, quantize_to_float8

os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


def quantize_model(part, quantization_scheme):
    if quantization_scheme == "int8":
        quantize_(part, int8_weight_only())
    elif quantization_scheme == "fp8":
        quantize_to_float8(part, QuantConfig(ActivationCasting.DYNAMIC))
    return part

class T2VModel():
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - quantization_scheme (str): The quantization scheme to use ('int8', 'fp8').
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    """
    def __init__(
        self,
        model_path: str,
        quantization_scheme: str = "fp8",
        dtype: torch.dtype = torch.bfloat16,
    ):
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder = quantize_model(part=text_encoder, quantization_scheme=quantization_scheme)
        transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
        transformer = quantize_model(part=transformer, quantization_scheme=quantization_scheme)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        vae = quantize_model(part=vae, quantization_scheme=quantization_scheme)
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype,
        )
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

        # Using with compile will run faster. First time infer will cost ~30min to compile.
        # self.pipe.transformer.to(memory_format=torch.channels_last)

        # for FP8 should remove self.pipe.enable_model_cpu_offload()
        self.pipe.enable_model_cpu_offload()

        # This is not for FP8 and INT8 and should remove this line
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_video(self, prompt, output_path, num_inference_steps, guidance_scale, num_videos_per_prompt):
        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=5,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        export_to_video(video, output_path, fps=8)

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
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./results/output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16', 'bfloat16')"
    )
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="bf16",
        choices=["int8", "fp8"],
        help="The quantization scheme to use (int8, fp8)",
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    check_prompt_args(args.prompt, args.prompt_file)
    
    print("Init model")
    start = time.time()
    model = T2VModel(
        model_path=args.model_path,
        quantization_scheme=args.quantization_scheme,
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
                    output_path=args.output_path,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                )
    else:
        raise ValueError("Either 'prompt' or 'prompt_file' must be provided")
    print(f"Video generation time: {time.time() - start:.2f}s")
    print("Done")