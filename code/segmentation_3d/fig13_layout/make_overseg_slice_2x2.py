from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# Settings
# =========================================================
WORKDIR = str(OUTPUT)
OUTPUT_NAME = "representative_overseg_slice_2x2.png"

NEW_SUPTITLE = (
    "Representative Over-segmentation Slice | "
    "BreaDM-Ma-1821 | Volume 2 (VIBRANT+C2) | Slice 2"
)

# 裁掉旧总标题，但保留四个小标题（MRI / Ground Truth / Prediction / Overlay）
TOP_CROP = 70

# 白底判断阈值
BG_THRESH = 245

# 布局参数
OUTER_PAD = 20
INNER_GAP_X = 36
INNER_GAP_Y = 36

# 总标题区域高度（加大一点）
TITLE_AREA_H = 70

# 总标题和下面四个小图标题之间额外留一点空隙
SUPTITLE_TO_PANEL_GAP = 10

# 总图标题字号（明显调大）
SUPTITLE_SIZE = 30

# =========================================================
# Helper functions
# =========================================================
def pick_source_png():
    source_dir = OUTPUT / "volume55_slice_overseg_selection"
    if not source_dir.exists():
        source_dir = RUN3 / "volume55_slice_overseg_selection"
    pngs = [p for p in glob.glob(str(source_dir / "*.png")) if os.path.basename(p) != OUTPUT_NAME]
    if not pngs:
        raise FileNotFoundError("当前目录没有找到可用的 PNG 图片。")
    # 优先选择体积最大的 png
    pngs = sorted(pngs, key=lambda p: os.path.getsize(p), reverse=True)
    return pngs[0]

def find_content_regions(binary_1d, min_width=40):
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(binary_1d):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            end = i - 1
            if end - start + 1 >= min_width:
                regions.append((start, end))
            in_region = False
    if in_region:
        end = len(binary_1d) - 1
        if end - start + 1 >= min_width:
            regions.append((start, end))
    return regions

def get_font(size):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()

# =========================================================
# Step 1. Read source image
# =========================================================
src_png = pick_source_png()
img = Image.open(src_png).convert("RGB")
arr = np.array(img)

print("=" * 70)
print("Source PNG:")
print(os.path.abspath(src_png))
print(f"Original size: {img.size}")
print("=" * 70)

# =========================================================
# Step 2. Remove old overall title strip
# =========================================================
if TOP_CROP >= arr.shape[0]:
    raise RuntimeError("TOP_CROP 设置过大，已经超过图像高度。")

content = arr[TOP_CROP:, :, :]

# =========================================================
# Step 3. Detect four column panels
# =========================================================
nonwhite = np.any(content < BG_THRESH, axis=2)

col_activity = nonwhite.sum(axis=0)
col_mask = col_activity > max(10, int(0.01 * nonwhite.shape[0]))
regions = find_content_regions(col_mask, min_width=40)

# 如果自动检测不是4块，则退化为等宽四分
if len(regions) != 4:
    print("[WARN] 自动检测到的列数不是 4，改用等宽四分。")
    h, w = content.shape[:2]
    step = w // 4
    regions = []
    for k in range(4):
        x1 = k * step
        x2 = (k + 1) * step - 1 if k < 3 else w - 1
        regions.append((x1, x2))

row_activity = nonwhite.sum(axis=1)
row_mask = row_activity > 10
row_idx = np.where(row_mask)[0]
if len(row_idx) == 0:
    raise RuntimeError("没有检测到内容区域。")

y1 = max(0, row_idx[0] - 8)
y2 = min(content.shape[0] - 1, row_idx[-1] + 8)

panels = []
for i, (x1, x2) in enumerate(regions, start=1):
    x1 = max(0, x1 - 8)
    x2 = min(content.shape[1] - 1, x2 + 8)
    crop = content[y1:y2+1, x1:x2+1, :]
    panels.append(Image.fromarray(crop))

if len(panels) != 4:
    raise RuntimeError(f"最终得到的面板数不是 4，而是 {len(panels)}。")

print("Detected 4 panels successfully.")
for i, p in enumerate(panels, start=1):
    print(f"Panel {i}: size = {p.size}")

# 顺序保持不变：
# 1 MRI, 2 GT, 3 Prediction, 4 Overlay
p1, p2, p3, p4 = panels

# =========================================================
# Step 4. Build 2x2 canvas
# =========================================================
left_col_w = max(p1.width, p3.width)
right_col_w = max(p2.width, p4.width)

top_row_h = max(p1.height, p2.height)
bottom_row_h = max(p3.height, p4.height)

canvas_w = OUTER_PAD + left_col_w + INNER_GAP_X + right_col_w + OUTER_PAD
canvas_h = (
    OUTER_PAD
    + TITLE_AREA_H
    + SUPTITLE_TO_PANEL_GAP
    + top_row_h
    + INNER_GAP_Y
    + bottom_row_h
    + OUTER_PAD
)

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

# 总标题
font_title = get_font(SUPTITLE_SIZE)
bbox = draw.textbbox((0, 0), NEW_SUPTITLE, font=font_title)
title_w = bbox[2] - bbox[0]
title_h = bbox[3] - bbox[1]
title_x = (canvas_w - title_w) // 2
title_y = OUTER_PAD + max(0, (TITLE_AREA_H - title_h) // 2) - 1
draw.text((title_x, title_y), NEW_SUPTITLE, fill="black", font=font_title)

# 四个 panel 的放置位置（2×2）
xL = OUTER_PAD
xR = OUTER_PAD + left_col_w + INNER_GAP_X
yTop = OUTER_PAD + TITLE_AREA_H + SUPTITLE_TO_PANEL_GAP
yBot = yTop + top_row_h + INNER_GAP_Y

def paste_center(base, panel, x, y, cell_w, cell_h):
    px = x + (cell_w - panel.width) // 2
    py = y + (cell_h - panel.height) // 2
    base.paste(panel, (px, py))

paste_center(canvas, p1, xL, yTop, left_col_w, top_row_h)
paste_center(canvas, p2, xR, yTop, right_col_w, top_row_h)
paste_center(canvas, p3, xL, yBot, left_col_w, bottom_row_h)
paste_center(canvas, p4, xR, yBot, right_col_w, bottom_row_h)

# =========================================================
# Step 5. Save
# =========================================================
out_path = os.path.join(WORKDIR, OUTPUT_NAME)
canvas.save(out_path, dpi=(300, 300))

print("=" * 70)
print("DONE")
print("Output:")
print(out_path)
print(f"Output size: {canvas.size}")
print("=" * 70)
