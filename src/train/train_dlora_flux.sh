export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="./data/instance"
export CLASS_DIR="./data/class"
export OUTPUT_DIR="./ckpt/ponix-generator"

accelerate launch src/train/train_dlora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=100 \
  --with_prior_preservation \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="photo of hcwf bird plush" \
  --class_prompt="photo of red bird plush, with 'POSTECH' logo on it's chest" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --gradient_checkpointing \
  --push_to_hub \
  --rank=16 \
  --cache_dir="/workspace/ponix-generator/model" \