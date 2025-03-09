from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file
from PIL import Image, ImageFilter
from pathlib import Path
import torch, os

prompt = """
photo of ponix <s0> plush bird traveling in outer space, 
orbiting planet Earth, highly detailed, hyper-realistic, 
cinematic lighting, intricate details, 8k resolution, 
ultra high quality photograph
"""

def run_dlora_advanced(lora_path:str, output_path:str):
    """
    inference code for dlora advanced (pivotal tuning, textual inversion)
    """
    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="/workspace/ponix-generator/model").to('cuda')
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

    text_encoders=[pipe.text_encoder, pipe.text_encoder_2]
    tokenizers=[pipe.tokenizer, pipe.tokenizer_2]

    state_dict = load_file("ckpt/ponix-generator/ponix-generator_emb.safetensors")
    pipe.load_textual_inversion(
        state_dict["clip_l"],
        token=["<s0>"],
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer
    )

    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
    ).images[0]

    image.save(output_path)


if __name__ == "__main__":
    run_dlora_advanced(
        lora_path="ckpt/ponix-generator/checkpoint-1500",
        output_path="ponix-generated.png"
    )