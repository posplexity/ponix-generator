export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="./data/instance-v0.2.0"
export OUTPUT_DIR="./ckpt/ponix-generator-v0.2.0"

accelerate launch src/train/train_dlora_flux_advanced.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --instance_data_dir "$INSTANCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mixed_precision "bf16" \
  --resolution 768 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --guidance_scale 1 \
  --optimizer "prodigy" \
  --learning_rate 1.0 \
  --max_train_steps 2500 \
  --gradient_checkpointing \
  --token_abstraction "ponix" \
  --instance_prompt "photo of ponix plush bird" \
  --initializer_concept "red plush bird" \
  --train_transformer_frac 1 \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac .25 \
  --rank 16 \
  --lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0" \
  --push_to_hub \
  --resume_from_checkpoint="latest" \
  --cache_dir="/workspace/ponix-generator/model" 
