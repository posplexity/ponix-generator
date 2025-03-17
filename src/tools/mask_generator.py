import gradio as gr
from PIL import Image

def save_mask(input_image_editor):
    """
    image_path: gr.ImageEditor(type='filepath')가 반환하는 최종 PNG 경로
    """
    mask_path = input_image_editor["layers"][0]

    if not mask_path:
        return "No image provided."

    # (1) 편집된 결과 이미지를 RGBA로 열기
    edited_img = Image.open(mask_path).convert("RGBA")

    # (2) 알파 채널(마스크) 추출
    alpha = edited_img.split()[-1]  # RGBA → A채널만
    # alpha>0 인 부분을 255(흰색), 아니면 0(검정)으로
    mask = alpha.point(lambda p: 255 if p > 0 else 0).convert("L")

    # (3) 마스크 파일로 저장
    mask.save("my_mask.png")
    return "Mask saved as 'my_mask.png'!"

with gr.Blocks() as demo:
    gr.Markdown("## Draw a mask on the image, then click 'Save Mask'")
    
    # 1) 이미지 업로드 + 마스크 스케치
    #    - type='filepath' → 내부적으로 편집 결과를 임시 PNG 경로로 반환
    #    - brush=gr.Brush(...) → 마우스로 흰색을 그릴 수 있음
    image_editor = gr.ImageEditor(
        label="Image Editor (draw your mask in white)",
        type="filepath",
        image_mode='RGBA',
        sources=["upload", "webcam"],
        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
        layers=False
    )

    # 2) 마스크 저장 버튼 & 결과 메시지
    btn_save = gr.Button("Save Mask")
    info_msg = gr.Textbox(label="Info")

    # 버튼 클릭 → save_mask 함수
    btn_save.click(fn=save_mask, inputs=image_editor, outputs=info_msg)

demo.launch()