import cv2
import numpy as np
from PIL import Image

W, H = 1024, 1024
canvas = np.zeros((H, W), dtype=np.uint8)

# 1. Weeping willow tree trunk on left (x=160..240, y=200..850)
cv2.line(canvas, (180, 850), (200, 520), 255, 6)
cv2.line(canvas, (220, 850), (210, 520), 255, 6)
# Branch forks
cv2.line(canvas, (205, 520), (140, 360), 255, 4)
cv2.line(canvas, (205, 520), (280, 340), 255, 4)
cv2.line(canvas, (205, 520), (360, 380), 255, 4)

# 2. Wooden park bench in foreground left-center (x=80..440, y=740..900)
# Backrest slats
bx1, by1 = 100, 750
bx2, by2 = 440, 850
for s in range(4):
    ys = by1 + s * 22
    cv2.line(canvas, (bx1, ys), (bx2, ys), 255, 4)
# Seat slats
for s in range(3):
    ys = by1 + 95 + s * 16
    cv2.line(canvas, (bx1 - 20, ys), (bx2 - 20, ys), 255, 4)
# Cast iron bench legs and armrests
cv2.line(canvas, (bx1, by1), (bx1 - 20, by1 + 95), 255, 5)
cv2.line(canvas, (bx1 - 20, by1 + 95), (bx1 - 30, by2 + 50), 255, 5)
cv2.line(canvas, (bx2, by1), (bx2 - 20, by1 + 95), 255, 5)
cv2.line(canvas, (bx2 - 20, by1 + 95), (bx2 - 10, by2 + 50), 255, 5)
cv2.line(canvas, (270, by1 + 10), (255, by1 + 95), 255, 4)
cv2.line(canvas, (255, by1 + 95), (250, by2 + 50), 255, 4)

# 3. Pond shoreline on right (x=480..1024, y=660..1024)
pond_contour = np.array([[480, 720], [560, 680], [740, 660], [920, 670], [1024, 680]], np.int32)
cv2.polylines(canvas, [pond_contour], False, 255, 4)
# Water ripples
for wy in range(720, 980, 45):
    cv2.line(canvas, (540, wy), (980, wy), 255, 1)

# Save clean sketch
sketch_img = Image.fromarray(canvas).convert("RGB")
sketch_img.save("/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/park_control_sketch.png")
print("Clean sketch created successfully!")
