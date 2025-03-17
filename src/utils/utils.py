import torch
import torch.nn.functional as F

def downsample_image_embeds(image_embeds: torch.FloatTensor, factor: int = 2, mode: str = "area"):
    """
    이미지 임베딩을 2D 그리드로 해석해 공간적으로 다운샘플링합니다.
    image_embeds.shape: (batch, tokens, dim), 여기서 tokens = m*m
    - factor: 다운샘플링 배수. (예: factor=2 → m -> m/2)
    - mode: torch.nn.functional.interpolate의 모드(예: 'area', 'bicubic', 'nearest' 등)
    """
    b, t, d = image_embeds.shape
    side = int(t**0.5)
    if side * side != t:
        raise ValueError(f"Cannot interpret {t} tokens as a square (m*m).")

    # (b, t, d) -> (b, m, m, d)
    image_embeds = image_embeds.view(b, side, side, d)
    # (b, m, m, d) -> (b, d, m, m)
    image_embeds = image_embeds.permute(0, 3, 1, 2)

    new_side = side // factor
    if new_side < 1:
        raise ValueError(f"Downsample factor {factor} is too large for side={side}.")

    # 실제 다운샘플
    image_embeds = F.interpolate(image_embeds, size=(new_side, new_side), mode=mode)

    # 다시 (b, new_side^2, d)로 되돌림
    image_embeds = image_embeds.permute(0, 2, 3, 1).reshape(b, new_side * new_side, d)
    return image_embeds


def patchify_mask(mask: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """
    ReduxAdvanced 노드에서처럼, 마스크를 patchify(14x14 등)해서
    CLIP 비전 임베딩과 동일한 2D 토큰 그리드 크기로 만든다.
    
    mask.shape가 (B, H, W) or (B, 1, H, W)라고 가정.
    return: (B, m, m), 여기서 m = H//patch_size (정사각형 가정)
    """
    if mask.dim() == 3:
        # (B, H, W) -> (B, 1, H, W)
        mask = mask.unsqueeze(1)
    b, c, h, w = mask.shape
    # maxPool로 patch 단위 downsample
    # 예: (H,W)=(224,224)이고 patch_size=14 => (224//14,224//14)=(16,16)
    # 각 (14x14) 블록 내에 1이라도 있으면 1, 없으면 0 (MaxPool2d)
    pool = torch.nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
    mask_patched = pool(mask)
    # (B,1,m,m) -> (B,m,m)
    mask_patched = (mask_patched > 0).float().squeeze(1)
    return mask_patched