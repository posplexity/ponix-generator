export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="./data/dreambooth/instance"
export CLASS_DIR="./data/dreambooth/class"
export OUTPUT_DIR="./ckpt/refinement-v0.0.2-flux-advanced"

poetry run accelerate launch train/train_dlora_flux_advanced.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --instance_data_dir "$INSTANCE_DIR" \
  --class_data_dir=$CLASS_DIR \
  --output_dir "$OUTPUT_DIR" \
  --mixed_precision "bf16" \
  --with_prior_preservation \
  --num_class_images 1000 \
  --resolution 768 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --optimizer "prodigy" \
  --learning_rate 1.0 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 2000 \
  --gradient_checkpointing \
  --token_abstraction "hcwf" \
  --instance_prompt "3d ultrasound image of hcwf baby face, headshot" \
  --class_prompt "3d ultrasound image of baby face, headshot" \
  --initializer_concept "3d ultrasound image of baby face, headshot" \
  --train_transformer_frac 1 \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac 0.5 \
  --lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0" \
  --seed "0" \
  --push_to_hub \
  --resume_from_checkpoint="latest" \
  --cache_dir="/workspace/ods-refinement-dev/model" \