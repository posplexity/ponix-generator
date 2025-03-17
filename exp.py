import torch
from diffusers import FluxPipeline, FluxInpaintPipeline
from src.flux_redux_pipeline import FluxPriorReduxPipeline
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    T5EncoderModel, 
    T5TokenizerFast
)
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# 1. 먼저 원하는 텍스트 인코더(클립, T5)와 토크나이저를 로드
clip_text_model = CLIPTextModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder", cache_dir="/workspace/ponix-generator/model", torch_dtype=torch.bfloat16)
clip_tokenizer = CLIPTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer", cache_dir="/workspace/ponix-generator/model")

t5_text_model = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", cache_dir="/workspace/ponix-generator/model", torch_dtype=torch.bfloat16)
t5_tokenizer = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2", cache_dir="/workspace/ponix-generator/model")

# GPU로 옮기기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_text_model = clip_text_model.to(device)
t5_text_model   = t5_text_model.to(device)

# 2. FluxPriorReduxPipeline 로드 시, text_encoder/text_encoder_2 등을 인자로 전달
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    pretrained_model_name_or_path="black-forest-labs/FLUX.1-Redux-dev",
    text_encoder=clip_text_model,      # 첫 번째 텍스트 인코더 (CLIP)
    tokenizer=clip_tokenizer,          # 첫 번째 토크나이저
    text_encoder_2=t5_text_model,      # 두 번째 텍스트 인코더 (T5)
    tokenizer_2=t5_tokenizer,          # 두 번째 토크나이저
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/ponix-generator/model",
).to(device)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev" , 
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/ponix-generator/model"
).to("cuda")

# 이제 `prompt=` 인자를 사용하는 예시
from diffusers.utils import load_image

image1 = load_image("./dog.jpg")
pipe_prior_output = pipe_prior_redux(
    image=image1,
    prompt="a cute dog on the beach",
    # negative_prompt="low quality, disfigured, watermark",
)

image2 = load_image("./dog.jpg")
mask = load_image("./my_mask.png")

pipe_prior_output2 = pipe_prior_redux(
    image=image2,
    mask=mask,
    # prompt="fine glass sculpture of a robot next to an eiffel tower",
    **pipe_prior_output
)

pipe_prior_output2.prompt_embeds = pipe_prior_output2.prompt_embeds.to(torch.bfloat16)
pipe_prior_output2.pooled_prompt_embeds = pipe_prior_output2.pooled_prompt_embeds.to(torch.bfloat16)

images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_prior_output2,
).images
images[0].save("flux-dev-redux.png")
