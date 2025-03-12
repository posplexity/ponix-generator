from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
from datetime import datetime
            
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="/workspace/ponix-generator/model").to('cuda')
pipeline.load_lora_weights('cwhuh/ponix-generator-v0.1.0', weight_name='pytorch_lora_weights.safetensors', subfolder="checkpoint-2500")
embedding_path = hf_hub_download(repo_id='cwhuh/ponix-generator-v0.1.0', filename='./ponix-generator-v0.1.0_emb.safetensors', repo_type="model")
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>", "<s2>"], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
            
prompt = """
photo of <s0><s1><s2> plush bird 
resting comfortably on a small raft of volcanic rock 
floating atop a glowing lava lake inside an erupting volcano, 
intense heat haze rising, 
embers and sparks flickering in the air, 
hyper-realistic details, cinematic lighting, 8k resolution, 
ultra high quality photograph, 
surreal, dramatic composition 
"""

# results 디렉토리 생성
os.makedirs("./results", exist_ok=True)

# 현재 시간을 파일명에 포함
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 여러 시드로 이미지 생성
seeds = range(1, 6)  # 1부터 5까지의 시드 사용
for seed in seeds:
    # 시드 설정
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # 이미지 생성
    image = pipeline(
        prompt,
        num_inference_steps=50,
        generator=generator
    ).images[0]
    
    # 결과 저장
    image.save(f"./results/ponix_generated_{timestamp}_seed{seed}.png")