import gradio as gr

# 이미지+마스크 처리를 위한 예시 함수
def process_image(image_data):
    # image_data는 ImageEditor로부터 받은 딕셔너리 (배경과 레이어 정보 포함)
    # 배경 이미지와 사용자가 그린 마스크를 분리합니다.
    base_image = image_data.get("image", image_data.get("background"))
    mask_image = None
    # layers 목록에서 첫 번째 레이어를 마스크로 사용 (배경 위에 그린 첫 마스크)
    if image_data.get("mask") is not None:
        mask_image = image_data["mask"]            # type="numpy"인 경우 키가 "mask"
    elif image_data.get("layers"):
        mask_image = image_data["layers"][0]       # type="pil"인 경우 첫 레이어 활용
    # ...여기서 base_image와 mask_image를 사용한 처리 로직 수행...
    return base_image  # (예시로, 처리 결과 대신 원본 이미지를 반환)

# Gradio 인터페이스 설정
image_editor = gr.ImageEditor(
    label="Input Image (Draw Mask)",
    sources=["upload"],        # 사용자 이미지 업로드 허용
    type="numpy",              # numpy 배열로 받음 (image_data에 'image'와 'mask' 키 생성)
    brush=gr.Brush(
        color_mode="fixed",    # 브러시 색상 고정
        colors=["#FFFFFF"],    # 흰색 브러시 (마스크 용)
        default_size=100       # 브러시 기본 반경 (픽셀)
    ),
    height=512
)
output_img = gr.Image(label="Output")

demo = gr.Interface(fn=process_image, inputs=image_editor, outputs=output_img)
demo.launch()