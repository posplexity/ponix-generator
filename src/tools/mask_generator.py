import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os

class MaskGeneratorGUI:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Mask Generator")

        # 원본 이미지를 Pillow로 열기
        self.original_image = Image.open(image_path).convert("RGB")
        self.width, self.height = self.original_image.size

        # 마스크를 (width, height) 크기의 8비트 회색(P=흑백) 이미지로 준비
        self.mask_image = Image.new("L", (self.width, self.height), 0)  
        # 0=검정(마스크 미적용), 255=흰색(마스크 적용)

        # GUI에 표시할 PIL 이미지 (원본 + 반투명 마스크)
        self.display_image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        # Canvas 생성
        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height, bg="gray")
        self.canvas.pack()

        # Canvas에 이미지 배치
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # 브러시 설정
        self.brush_size = 20
        self.drawing_color = 255  # 마스크를 칠할 때 255(흰색)
        self.eraser_mode = False  # eraser mode가 켜지면 drawing_color=0으로 동작

        # 마스크에 직접 그리기 위한 ImageDraw 객체
        self.draw_mask = ImageDraw.Draw(self.mask_image)

        # 마우스 이벤트 바인딩
        self.canvas.bind("<B1-Motion>", self.paint)       # 왼쪽 버튼 드래그
        self.canvas.bind("<ButtonPress-1>", self.paint)   # 왼쪽 버튼 처음 클릭

        # 하단 툴박스(브러시 사이즈, 지우개 모드, 저장 등)
        self.toolbar = tk.Frame(self.master)
        self.toolbar.pack(fill=tk.X, pady=5)

        tk.Label(self.toolbar, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_scale = tk.Scale(self.toolbar, from_=1, to=100,
                                    orient=tk.HORIZONTAL, command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.eraser_button = tk.Button(self.toolbar, text="Eraser OFF", command=self.toggle_eraser)
        self.eraser_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.toolbar, text="Clear Mask", command=self.clear_mask)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.toolbar, text="Save & Exit", command=self.save_and_exit)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def update_brush_size(self, val):
        self.brush_size = int(val)

    def toggle_eraser(self):
        # Eraser모드 토글
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.drawing_color = 0   # 지우개일 땐 마스크를 0(검정)으로
            self.eraser_button.config(text="Eraser ON")
        else:
            self.drawing_color = 255
            self.eraser_button.config(text="Eraser OFF")

    def paint(self, event):
        # (x, y) 좌표
        x = event.x
        y = event.y
        r = self.brush_size // 2

        # 마스크 이미지에 그리기 (원 형태)
        self.draw_mask.ellipse([x-r, y-r, x+r, y+r], fill=self.drawing_color)

        # UI 표시용 (투명도 있는 빨간색 브러시 등으로 덮어씌우고 싶으면 추가 로직 가능)
        # 여기서는 단순히 기존 display_image에 반투명 덧칠
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        # 마스크는 흰색=불투명 / 검정=아무것도 안 칠함 정도로 표현 가능
        brush_color = (255, 0, 0, 128) if self.drawing_color == 255 else (0, 0, 0, 128)
        draw_overlay.ellipse([x-r, y-r, x+r, y+r], fill=brush_color)

        # 합성
        self.display_image = self.original_image.copy().convert("RGBA")
        self.display_image.alpha_composite(overlay)

        # 갱신
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

    def clear_mask(self):
        # 마스크 전체를 검정(0)으로 채우고 다시 표시
        self.draw_mask.rectangle([0, 0, self.width, self.height], fill=0)
        self.display_image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

    def save_and_exit(self):
        # 마스크 저장 (PNG로)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files","*.png"), ("All Files","*.*")],
            title="Save Mask"
        )
        if save_path:
            # self.mask_image: L 모드(0-255)
            # 255=마스크 영역, 0=비마스크
            self.mask_image.save(save_path)
            print(f"Mask saved at {save_path}")
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()

    # 1) 이미지 파일 선택 창을 띄우고 싶다면:
    # img_path = filedialog.askopenfilename(
    #     title="Select Image",
    #     filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.gif"), ("All files", "*.*")]
    # )

    # 2) 혹은 직접 경로 지정
    img_path = "example.jpg"  # <--- 수정
    if not os.path.exists(img_path):
        print("Error: example.jpg not found. Please specify a valid image path.")
    else:
        app = MaskGeneratorGUI(root, img_path)
        root.mainloop()