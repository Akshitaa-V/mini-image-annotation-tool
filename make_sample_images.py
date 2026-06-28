"""
Generates a handful of simple synthetic images that stand in for photographed
recyclable objects (so the annotation tool has something to annotate without
needing internet access or a real camera). Swap this for your own phone
photos by just pointing annotate.py at a different folder.
"""
import os
import numpy as np
import cv2

OUT_DIR = "sample_images"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

def blank_canvas(w=640, h=480, bg=(235, 235, 235)):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    return img

def add_noise(img, strength=6):
    noise = rng.normal(0, strength, img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

# Image 1: a "bottle" (tall rectangle + cap) and a "can" (shorter rectangle)
img1 = blank_canvas()
cv2.rectangle(img1, (120, 100), (220, 400), (40, 120, 200), -1)   # bottle body (PET-ish blue)
cv2.rectangle(img1, (150, 70), (190, 100), (40, 120, 200), -1)    # bottle cap area
cv2.rectangle(img1, (350, 220), (470, 400), (180, 180, 180), -1)  # can body (metal-ish grey)
cv2.imwrite(os.path.join(OUT_DIR, "sample_01.jpg"), add_noise(img1))

# Image 2: a "cardboard box" (brown square) and a "battery" (small dark rectangle)
img2 = blank_canvas()
cv2.rectangle(img2, (80, 150), (320, 380), (60, 100, 150), -1)    # cardboard box (brownish)
cv2.rectangle(img2, (420, 250), (470, 340), (30, 30, 30), -1)     # battery (dark)
cv2.imwrite(os.path.join(OUT_DIR, "sample_02.jpg"), add_noise(img2))

# Image 3: two "cans" of different sizes (mixed metal waste)
img3 = blank_canvas()
cv2.rectangle(img3, (100, 200), (200, 400), (170, 170, 170), -1)
cv2.rectangle(img3, (350, 240), (430, 400), (190, 190, 190), -1)
cv2.imwrite(os.path.join(OUT_DIR, "sample_03.jpg"), add_noise(img3))

print(f"Generated 3 sample images in ./{OUT_DIR}/")
