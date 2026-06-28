"""
Mini Image Annotation Tool
---------------------------
A small, fully local, no-dependency-beyond-OpenCV tool for drawing bounding
boxes on images and saving labels in YOLO format.

USAGE (run on your own machine, not in a headless sandbox):
    python annotate.py --images sample_images --classes classes.txt --out labels

CONTROLS:
    - Left-click and drag to draw a box
    - Press a number key (0-9) to assign the box you just drew to that class index
    - 'u' = undo last box
    - 's' = save annotations for current image and move to next
    - 'n' = skip to next image without saving
    - 'q' = quit

OUTPUT FORMAT (YOLO style, one .txt per image):
    <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    All values normalized to [0, 1] relative to image width/height.
"""
import argparse
import os
import cv2

class Annotator:
    def __init__(self, image_paths, class_names, out_dir):
        self.image_paths = image_paths
        self.class_names = class_names
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.idx = 0
        self.boxes = []  # list of (class_id, x1, y1, x2, y2) in pixel coords
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.img = None
        self.display_img = None

    def load_image(self):
        path = self.image_paths[self.idx]
        self.img = cv2.imread(path)
        if self.img is None:
            raise FileNotFoundError(f"Could not read {path}")
        self.boxes = []
        self.current_box = None
        self.redraw()

    def redraw(self):
        self.display_img = self.img.copy()
        for cls_id, x1, y1, x2, y2 in self.boxes:
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            cv2.rectangle(self.display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display_img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1, y1 = self.start_point
            self.current_box = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            self.redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_box is not None:
                x1, y1, x2, y2 = self.current_box
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                    # Default to class 0 until the user presses a number key
                    self.boxes.append((0, x1, y1, x2, y2))
                self.current_box = None
                self.redraw()

    def save_labels(self):
        path = self.image_paths[self.idx]
        h, w = self.img.shape[:2]
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(self.out_dir, f"{stem}.txt")
        with open(out_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Saved {len(self.boxes)} box(es) -> {out_path}")

    def run(self):
        window = "Mini Annotator (u=undo, 0-9=class of last box, s=save+next, n=skip, q=quit)"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.mouse_callback)
        self.load_image()

        while self.idx < len(self.image_paths):
            cv2.imshow(window, self.display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('u') and self.boxes:
                self.boxes.pop()
                self.redraw()
            elif key == ord('s'):
                self.save_labels()
                self.idx += 1
                if self.idx < len(self.image_paths):
                    self.load_image()
            elif key == ord('n'):
                self.idx += 1
                if self.idx < len(self.image_paths):
                    self.load_image()
            elif ord('0') <= key <= ord('9') and self.boxes:
                cls_id = key - ord('0')
                last = self.boxes[-1]
                self.boxes[-1] = (cls_id, last[1], last[2], last[3], last[4])
                self.redraw()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder of images to annotate")
    parser.add_argument("--classes", required=True, help="Text file, one class name per line")
    parser.add_argument("--out", default="labels", help="Output folder for YOLO-format .txt labels")
    args = parser.parse_args()

    image_paths = sorted(
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    with open(args.classes) as f:
        class_names = [line.strip() for line in f if line.strip()]

    Annotator(image_paths, class_names, args.out).run()


if __name__ == "__main__":
    main()
