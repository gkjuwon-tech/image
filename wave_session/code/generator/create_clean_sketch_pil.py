import math
from PIL import Image, ImageDraw

W, H = 1024, 1024
img = Image.new("RGB", (W, H), (0, 0, 0))
draw = ImageDraw.Draw(img)

# 1. Weeping willow tree trunk on far left (x=60..120, y=180..750)
draw.line([(80, 750), (95, 450)], fill=(255, 255, 255), width=6)
draw.line([(120, 750), (105, 450)], fill=(255, 255, 255), width=6)
# Canopy main branches arching overhead
draw.line([(100, 450), (220, 320)], fill=(255, 255, 255), width=5)
draw.line([(220, 320), (450, 260)], fill=(255, 255, 255), width=4)
draw.line([(100, 450), (70, 280)], fill=(255, 255, 255), width=4)
# Weeping willow cascading leafy vines
for vx in range(80, 520, 24):
    bend = int(18 * math.sin(vx * 0.04))
    draw.line([(vx, 260), (vx + bend, 500)], fill=(255, 255, 255), width=2)

# 2. Wooden park bench in foreground left-center (x=160..480, y=720..880)
bx1, by1 = 180, 720
bx2, by2 = 500, 810
# Backrest slats
for s in range(4):
    ys = by1 + s * 20
    draw.line([(bx1, ys), (bx2, ys)], fill=(255, 255, 255), width=4)
# Seat slats
for s in range(3):
    ys = by1 + 90 + s * 16
    draw.line([(bx1 - 20, ys), (bx2 - 20, ys)], fill=(255, 255, 255), width=4)
# Bench legs and cast-iron armrests
draw.line([(bx1, by1), (bx1 - 20, by1 + 90), (bx1 - 30, by2 + 50)], fill=(255, 255, 255), width=5)
draw.line([(bx2, by1), (bx2 - 20, by1 + 90), (bx2 - 10, by2 + 50)], fill=(255, 255, 255), width=5)
draw.line([(340, by1 + 10), (325, by1 + 90), (320, by2 + 50)], fill=(255, 255, 255), width=4)

# 3. Pond shoreline on right (x=540..1024, y=660..1024)
pond_pts = [(540, 720), (620, 680), (780, 660), (940, 670), (1024, 680)]
draw.line(pond_pts, fill=(255, 255, 255), width=4)
# Gentle water ripples
for wy in range(710, 990, 40):
    draw.line([(600, wy), (980, wy)], fill=(255, 255, 255), width=1)

img.save("/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/park_control_sketch.png")
print("Natural non-overlapping sketch created!")
