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
embedding_path = hf_hub_download(repo_id='cwhuh/ponix-generator-v0.2.0', filename='./ponix-generator-v0.2.0_emb.safetensors', repo_type="model")
state_dict = load_file(embedding_path)
pipe_prior_redux.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>", "<s2>"], text_encoder=pipe_prior_redux.text_encoder, tokenizer=pipe_prior_redux.tokenizer)


pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev" , 
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16,
    cache_dir="/workspace/ponix-generator/model"
).to("cuda")
pipe.load_lora_weights('cwhuh/ponix-generator-v0.2.0', weight_name='pytorch_lora_weights.safetensors')

# 이제 `prompt=` 인자를 사용하는 예시
from diffusers.utils import load_image

image1 = load_image("./data/img/ponix.webp")
mask1 = load_image("./data/img/ponix_mask.png")
pipe_prior_output = pipe_prior_redux(
    image=image1,
    mask=mask1,
    prompt="photo of <s0><s1><s2> plush bird wearing korean traditional clothes in front of a building",
    # negative_prompt="low quality, disfigured, watermark",
)

image2 = load_image("./data/img/hanbok.webp")
mask2 = load_image("./data/img/hanbok_mask.png")
pipe_prior_output2 = pipe_prior_redux(
    image=image2,
    mask=mask2,
    downsample_factor=2,
    **pipe_prior_output
)

image3 = load_image("./data/img/changeup.jpeg")
pipe_prior_output3 = pipe_prior_redux(
    image=image3,
    **pipe_prior_output2
)


pipe_prior_output3.prompt_embeds = pipe_prior_output3.prompt_embeds.to(torch.bfloat16)
pipe_prior_output3.pooled_prompt_embeds = pipe_prior_output3.pooled_prompt_embeds.to(torch.bfloat16)

images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_prior_output3,
).images
images[0].save("flux-dev-redux.png")
