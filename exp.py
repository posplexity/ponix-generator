from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
            
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="/workspace/ponix-generator/model").to('cuda')
pipeline.load_lora_weights('cwhuh/ponix-generator-v0.1.0', weight_name='pytorch_lora_weights.safetensors')
embedding_path = hf_hub_download(repo_id='cwhuh/ponix-generator-v0.1.0', filename='./ponix-generator-v0.1.0_emb.safetensors', repo_type="model")
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>", "<s2>"], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
            
image = pipeline('photo of <s0><s1><s2> plush bird and astronaut sitting on an old bench, in front of dark prison', num_inference_steps=50).images[0]
image.save("ponix-generated.png")