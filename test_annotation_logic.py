"""
Headless validation of the annotation tool's core logic: given a box drawn
in pixel coordinates, does it save correctly as normalized YOLO format and
read back to the same pixel box? This is the same math used inside
annotate.py's save_labels(), tested without needing a display.
"""
import os
import cv2

from annotate import Annotator

IMG_DIR = "sample_images"
OUT_DIR = "labels_test"
CLASSES = ["bottle", "can", "cardboard", "battery"]

image_paths = sorted(
    os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")
)

ann = Annotator(image_paths, CLASSES, OUT_DIR)

# Simulate annotating sample_01.jpg: one "bottle" box, one "can" box
ann.load_image()
h, w = ann.img.shape[:2]
ann.boxes = [
    (0, 120, 100, 220, 400),  # class 0 = bottle, matches the synthetic bottle region
    (1, 350, 220, 470, 400),  # class 1 = can, matches the synthetic can region
]
ann.save_labels()

# Read the saved file back and reconstruct pixel boxes to verify round-trip correctness
out_path = os.path.join(OUT_DIR, "sample_01.txt")
with open(out_path) as f:
    lines = [line.split() for line in f if line.strip()]

print(f"\nImage size: {w}x{h}")
print(f"Saved {len(lines)} line(s) to {out_path}:\n")

ok = True
expected = [(0, 120, 100, 220, 400), (1, 350, 220, 470, 400)]
for line, (exp_cls, ex1, ey1, ex2, ey2) in zip(lines, expected):
    cls_id, xc, yc, bw, bh = int(line[0]), *map(float, line[1:])
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    print(f"  class={cls_id} -> reconstructed box ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  "
          f"expected ({ex1}, {ey1}, {ex2}, {ey2})")
    if cls_id != exp_cls or abs(x1 - ex1) > 1 or abs(y1 - ey1) > 1 or abs(x2 - ex2) > 1 or abs(y2 - ey2) > 1:
        ok = False

print("\nROUND-TRIP TEST:", "PASSED" if ok else "FAILED")

# Draw the boxes onto the image and save a visual proof for inspection
vis = ann.img.copy()
for cls_id, x1, y1, x2, y2 in ann.boxes:
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, CLASSES[cls_id], (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite("annotation_preview.jpg", vis)
print("Visual proof saved -> annotation_preview.jpg")
