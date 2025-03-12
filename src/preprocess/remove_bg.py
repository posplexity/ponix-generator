from transformers import AutoModelForImageSegmentation
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path
from torchvision import transforms

# 모델 로드
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)

# 입력/출력 디렉토리
input_dir = "./data/raw-v0.2.0-front"
output_dir = "./data/instance-v0.2.0-front"
os.makedirs(output_dir, exist_ok=True)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
torch.set_float32_matmul_precision('high')
model.eval()

# 이미지 변환 (모델 입력용)
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def remove_background(image_path, output_path):
    # 원본 이미지 로드 (색상 보존)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, 3), uint8

    # 모델 입력용 변환
    input_image = transform_image(img).unsqueeze(0).to(device)

    # 추론 (마스크 예측)
    with torch.no_grad():
        # RMBG 모델 출력: (batch, channels=1, h, w) 형태 목록
        pred = model(input_image)[-1].sigmoid().cpu()  # 0~1 범위

    # pred.shape = (1, 1, 1024, 1024)
    mask = pred[0].squeeze(0)  # (1024, 1024)

    # 마스크를 PIL로 변환 (0~1 → 0~255)
    mask_pil = transforms.ToPILImage()(mask)  
    # 원본 이미지 크기에 맞춰 리사이즈
    mask_resized = mask_pil.resize(img.size, resample=Image.BILINEAR)  
    mask_np = np.array(mask_resized, dtype=np.float32) / 255.0  # 이제 0~1 float 범위

    # 흰색 배경 생성
    white_bg = np.ones_like(img_np, dtype=np.uint8) * 255  # (H, W, 3)

    # 마스크를 3채널로 확장
    mask_np_3ch = np.stack([mask_np, mask_np, mask_np], axis=2)  # (H, W, 3), float32

    # 알파 블렌딩 (원본 * 마스크 + 흰 배경 * (1 - 마스크))
    img_float = img_np.astype(np.float32)
    white_bg_float = white_bg.astype(np.float32)

    result_float = img_float * mask_np_3ch + white_bg_float * (1.0 - mask_np_3ch)
    result = np.clip(result_float, 0, 255).astype(np.uint8)

    # 결과를 PIL 이미지로 변환
    result_img = Image.fromarray(result)
    
    # 이미지를 오른쪽으로 90도 회전 (시계 방향)
    rotated_img = result_img.rotate(-90, expand=True)  # PIL에서는 반시계 방향이 양수, 시계 방향이 음수
    
    # 회전된 이미지 저장
    rotated_img.save(output_path)

# 실행
image_files = list(Path(input_dir).glob("*.jpg")) \
             + list(Path(input_dir).glob("*.png")) \
             + list(Path(input_dir).glob("*.jpeg"))

print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

for img_path in tqdm(image_files):
    output_path = os.path.join(output_dir, img_path.name)
    try:
        remove_background(img_path, output_path)
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {img_path}, 오류: {e}")

print(f"모든 이미지 처리가 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")