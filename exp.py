import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

pipe: FluxPipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/ponix-generator/model"
).to("cuda")

image = load_image("monalisa.png").resize((1024, 1024))

pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter-v2", weight_name="ip_adapter.safetensors", image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")

def LinearStrengthModel(start, finish, size):
    return [
        (start + (finish - start) * (i / (size - 1))) for i in range(size)
    ]

ip_strengths = LinearStrengthModel(0.4, 1.0, 19)
pipe.set_ip_adapter_scale(ip_strengths)

image = pipe(
    width=1024,
    height=1024,
    prompt='wearing red sunglasses, golden chain and a green cap',
    negative_prompt="",
    true_cfg_scale=1.0,
    generator=torch.Generator().manual_seed(0),
    ip_adapter_image=image,
).images[0]

image.save('result.jpg')