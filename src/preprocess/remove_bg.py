# Load model directly
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path
from torchvision import transforms

# 입력 및 출력 디렉토리 설정
input_dir = "./data/instance-v0.1.0"
output_dir = "./data/instance-v0.1.1"

# 출력 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 연산 정밀도 설정 (성능 향상)
torch.set_float32_matmul_precision('high')
model.eval()

# 이미지 변환 설정
image_size = (1024, 1024)  # 고정된 이미지 크기 설정
transform_image = transforms.Compose([
    transforms.Resize(image_size),  # 모든 이미지를 동일한 크기로 리사이징
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image_path, output_path):
    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    
    # 이미지 변환
    input_image = transform_image(img).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        pred = model(input_image)[-1].sigmoid().cpu()
    
    # 마스크 추출
    mask = pred[0].squeeze()
    mask_pil = transforms.ToPILImage()(mask)
    mask_resized = mask_pil.resize(img.size)
    
    # 마스크를 numpy 배열로 변환
    mask_np = np.array(mask_resized)
    
    # 원본 이미지를 numpy 배열로 변환
    img_np = np.array(img)
    
    # 흰색 배경 생성
    white_bg = np.ones_like(img_np) * 255
    
    # 마스크를 3채널로 확장 (RGB 각 채널에 적용하기 위해)
    mask_np_3channel = np.stack([mask_np] * 3, axis=2)
    
    # 마스크를 사용하여 원본 이미지와 흰색 배경 합성
    # 마스크 값이 높을수록 (객체에 가까울수록) 원본 이미지 픽셀 사용
    # 마스크 값이 낮을수록 (배경에 가까울수록) 흰색 배경 픽셀 사용
    result = img_np * mask_np_3channel + white_bg * (1 - mask_np_3channel)
    result = result.astype(np.uint8)
    
    # numpy 배열을 PIL 이미지로 변환
    result_img = Image.fromarray(result)
    
    # 저장
    result_img.save(output_path)

# 입력 디렉토리의 모든 이미지 처리
image_files = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpeg"))

print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

for img_path in tqdm(image_files):
    # 출력 파일 경로 생성
    output_path = os.path.join(output_dir, img_path.name)
    
    # 배경 제거 및 저장
    try:
        remove_background(img_path, output_path)
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {img_path}, 오류: {e}")

print(f"모든 이미지 처리가 완료되었습니다. 결과는 {output_dir}에 저장되었습니다.")
