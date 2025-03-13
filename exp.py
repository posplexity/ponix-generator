from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
from datetime import datetime
            
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="/workspace/ponix-generator/model").to('cuda')
<<<<<<< Updated upstream
pipeline.load_lora_weights('cwhuh/ponix-generator-v0.2.0', weight_name='pytorch_lora_weights.safetensors')
embedding_path = hf_hub_download(repo_id='cwhuh/ponix-generator-v0.2.0', filename='./ponix-generator-v0.2.0_emb.safetensors', repo_type="model")
=======
pipeline.load_lora_weights('cwhuh/ponix-generator-v0.1.0', weight_name='pytorch_lora_weights.safetensors', subfolder="checkpoint-4000")
embedding_path = hf_hub_download(repo_id='cwhuh/ponix-generator-v0.1.0', filename='ponix-generator-v0.1.0_emb.safetensors', repo_type="model")
>>>>>>> Stashed changes
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>", "<s2>"], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
            
prompt = """
photo of <s0><s1><s2> plush bird 
<<<<<<< Updated upstream
swimming in the ocean, 
gentle waves surrounding it, 
crystal clear blue water, 
sunlight reflecting off the water surface, 
hyper-realistic details, cinematic lighting, 8k resolution, 
ultra high quality photograph, 
serene, natural composition
=======
wearing a graduation cap and casual collegiate outfit
inside a prestigious university campus on orientation day, 
surrounded by historic academic buildings and green quads,
carrying a backpack filled with textbooks,
other freshman students visible in background,
campus library and lecture halls visible in the scene,
hyper-realistic details, bright sunny day lighting, 8k resolution, 
ultra high quality photograph, 
academic atmosphere, excitement of new beginnings
>>>>>>> Stashed changes
"""

# results 디렉토리 생성
os.makedirs("./results", exist_ok=True)

# 현재 시간을 파일명에 포함
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 여러 시드로 이미지 생성
seeds = range(21, 41)  # 1부터 5까지의 시드 사용
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